import pickle
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from psbody.mesh import Mesh


from reconstruction import AE, run, eval_error
from utils import utils, writer, DataLoader, mesh_sampling

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='interpolation_exp')
parser.add_argument('--dataset', type=str, default='CoMA')
parser.add_argument('--split', type=str, default='interpolation')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# network hyperparameters
parser.add_argument('--out_channels',
                    nargs='+',
                    default=[32, 32, 32, 64],
                    type=int)
parser.add_argument('--latent_channels', type=int, default=16)
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0)

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=300)

# others
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, '..', 'data', args.dataset)
args.out_dir = osp.join(args.work_dir, 'out', args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer.Writer(args)
device = torch.device('cuda', args.device_idx)
torch.set_num_threads(args.n_threads)

# deterministic
torch.manual_seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

# load dataset
template_fp = osp.join(args.data_fp, 'template', 'template.obj')

if args.dataset == 'CoMA':
    from datasets import MeshData
    meshdata = MeshData(args.data_fp,
                        template_fp,
                        split=args.split,
                        test_exp=args.test_exp)
    train_loader = DataLoader(meshdata.train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)
    test_loader = DataLoader(meshdata.test_dataset, batch_size=args.batch_size)

elif args.dataset == 'ThreeDFN':
    from datasets import ThreeDFN
    # Load 3DFN dataset (automatically loads train/test split)
    train_dataset = ThreeDFN(root=args.data_fp, train=True, transform=T.NormalizeScale())
    test_dataset = ThreeDFN(root=args.data_fp, train=False, transform=T.NormalizeScale())

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# generate/load transform matrices
transform_fp = osp.join(args.data_fp, 'transform.pkl')
if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)
    ds_factors = [4, 4, 4, 4]
    _, A, D, U, F, V = mesh_sampling.generate_transform_matrices(
        mesh, ds_factors)
    tmp = {
        'vertices': V,
        'face': F,
        'adj': A,
        'down_transform': D,
        'up_transform': U
    }

    with open(transform_fp, 'wb') as fp:
        pickle.dump(tmp, fp)
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    with open(transform_fp, 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

spiral_indices_list = [
    utils.preprocess_spiral(tmp['face'][idx], args.seq_length[idx],
                            tmp['vertices'][idx],
                            args.dilation[idx]).to(device)
    for idx in range(len(tmp['face']) - 1)
]
down_transform_list = [
    utils.to_sparse(down_transform).to(device)
    for down_transform in tmp['down_transform']
]
up_transform_list = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

model = AE(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list,
           up_transform_list).to(device)
print('Number of parameters: {}'.format(utils.count_parameters(model)))
print(model)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            args.decay_step,
                                            gamma=args.lr_decay)

run(model, train_loader, test_loader, args.epochs, optimizer, scheduler,
    writer, device)

if args.dataset == 'CoMA':
    eval_error(model, test_loader, device, meshdata, args.out_dir)
elif args.dataset == 'ThreeDFN':
    eval_error(model, test_loader, device, test_dataset, args.out_dir)

def reverse_preprocessing(data, dataset):
    """
    Reverse normalization using dataset's mean and std
    """
    device = data.device  # Get the device of `data`
    mean = dataset.global_mean.to(device)
    std = dataset.global_std.to(device)
    
    return (data * std) + mean  # Now all tensors are on the same device

def infer_and_save(test_loader, model, device, dataset, output_dir, template_mesh_path):
    """
    Run inference on the test dataset and save predictions as .obj files using input file names.

    Parameters:
    - test_loader (DataLoader): DataLoader for the test set.
    - model (torch.nn.Module): The trained model for inference.
    - device (torch.device): Device to run the inference on.
    - dataset (InMemoryDataset): The dataset to reverse the preprocessing (mean and std).
    - output_dir (str): Directory to save the output mesh files.
    - template_mesh_path (str): Path to the template mesh to extract faces.
    """
    model.eval()
    predictions = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)  # Move data to the correct device
            pred = model(data.x)    # Forward pass to get model predictions
            pred = reverse_preprocessing(pred, dataset)  # Reverse preprocessing

            # Loop through each element in the batch and save individual predictions
            for i, filename in enumerate(data.filename):  # data.filename is a list of filenames in the batch
                pred_single = pred[i]  # Get the prediction for this element in the batch
                predictions.append((pred_single, filename))  # Save with corresponding filename
    
    # Save predictions as meshes
    save_predictions(predictions, output_dir, template_mesh_path)

def save_predictions(predictions, output_dir, template_mesh_path):
    """
    Saves the model's predictions as mesh files (.obj) using input file names.

    Parameters:
    - predictions (list of tuples): Each tuple contains (tensor prediction, input filename).
    - output_dir (str): Directory to save the output .obj files.
    - template_mesh_path (str): Path to the template mesh to extract faces.
    """
    template_mesh = Mesh(filename=template_mesh_path)  # Load template mesh

    for pred, filename in predictions:
        pred_np = pred.detach().cpu().numpy()  # Ensure it's a NumPy array
        pred_np = pred_np.reshape(-1, 3)  # Ensure correct shape

        output_filename = filename.replace('.obj', '_pred.obj')  # Replace .obj with _pred.obj
        mesh = Mesh(v=pred_np, f=template_mesh.f)  # Use template faces
        mesh.write_obj(os.path.join(output_dir, output_filename))

output_dir = osp.join(args.data_fp, 'predicted')  # Output directory to save the predictions

# Run inference and save predictions
infer_and_save(test_loader, model, device, test_dataset, output_dir, template_fp)

def get_latent_embeddings(test_loader, model, device, data_fp):
    model.eval()
    path = osp.join(data_fp, 'latent_embeddings.pt')
    latent_embeddings = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            latent = model.encoder(data.x)  # Extract latent embeddings
            latent_embeddings.append(latent)
    
    # Convert list to tensor
    latent_embeddings = torch.cat(latent_embeddings, dim=0)
    
    # Save the embeddings to a file
    torch.save(latent_embeddings, path)

    return latent_embeddings

latent_embeddings = get_latent_embeddings(test_loader, model, device, args.data_fp)