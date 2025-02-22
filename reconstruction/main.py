import pickle
import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_geometric.transforms as T
from psbody.mesh import Mesh
import openmesh as om


from reconstruction import AEVAE, run, eval_error
from utils import utils, writer, DataLoader, mesh_sampling

parser = argparse.ArgumentParser(description='mesh autoencoder')
parser.add_argument('--exp_name', type=str, default='autoencoder_3dfn')
parser.add_argument('--dataset', type=str, default='ThreeDFN', choices=['CoMA', 'foundation', 'ThreeDFN'])
parser.add_argument('--split', type=str, default='interpolation')
parser.add_argument('--test_exp', type=str, default='bareteeth')
parser.add_argument('--n_threads', type=int, default=4)
parser.add_argument('--device_idx', type=int, default=0)

# network hyperparameters
parser.add_argument('--use_vae', action='store_true')
parser.add_argument('--out_channels', nargs='+', default=[32, 64, 128, 128], type=int)
parser.add_argument('--latent_channels', type=int, default=32)  # Larger latent space for better embeddings
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--seq_length', type=int, default=[9, 9, 9, 9], nargs='+')
parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')

# optimizer hyperparmeters
parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'rmsprop'])
parser.add_argument('--scheduler', type=str, default='cosine', choices=['onecycle', 'cosine'])
parser.add_argument('--lr', type=float, default=3e-4)  # Lower initial LR for stable training
parser.add_argument('--lr_decay', type=float, default=0.99)
parser.add_argument('--decay_step', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=1e-4)  # Prevents overfitting in latent space
parser.add_argument('--beta', type=float, default=1.0)  # Higher beta → more disentanglement

# training hyperparameters
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=300)

# others
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file to resume training or evaluate')
parser.add_argument('--resume', action='store_true', help='Resume training from the checkpoint')

args = parser.parse_args()

args.work_dir = osp.dirname(osp.realpath(__file__))
args.data_fp = osp.join(args.work_dir, '..', 'data', args.dataset)
args.out_dir = osp.join(args.work_dir, 'out', args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
print(args)

utils.makedirs(args.out_dir)
utils.makedirs(args.checkpoints_dir)

writer = writer.Writer(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

elif args.dataset == 'foundation':
    from datasets import ThreeDFN
    # Load foundation dataset (automatically loads train/test split)
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

model = AEVAE(args.in_channels, args.out_channels, args.latent_channels,
           spiral_indices_list, down_transform_list,
           up_transform_list, args.use_vae).to(device)
print('Number of parameters: {}'.format(utils.count_parameters(model)))
print(model)

if args.optimizer.lower() == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
elif args.optimizer.lower() == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
elif args.optimizer.lower() == 'rmsprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.scheduler.lower() == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
elif args.scheduler.lower() == 'plateau':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
elif args.scheduler.lower() == 'onecycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr * 10, total_steps=args.epochs * len(train_loader), pct_start=0.3)
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.lr_decay)

# Load checkpoint if provided
start_epoch = 0
if args.checkpoint is not None:
  if osp.isfile(args.checkpoint):
      print(f"Loading checkpoint from {args.checkpoint}")
      checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
      print(f"Loading model state dict from checkpoint")
      model.load_state_dict(checkpoint['model_state_dict'])
      if args.resume:
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
          start_epoch = checkpoint.get('epoch', 0)
          print(f"Resuming training from epoch {start_epoch}")
  else:
      print("Faulty checkpoint path, args.checkpoint is: ", args.checkpoint)

# Determine if training should proceed
if args.resume or args.checkpoint is None:
    remaining_epochs = args.epochs - start_epoch
    if remaining_epochs > 0:
        run(model, train_loader, test_loader, remaining_epochs, optimizer, scheduler,
            writer, device, args, start_epoch, args.beta)
    else:
        print("Training already completed. Skipping training phase.")

if args.dataset == 'CoMA':
    eval_error(model, test_loader, device, meshdata, args.out_dir)
elif args.dataset == 'ThreeDFN' or args.dataset == 'foundation':
    eval_error(model, test_loader, device, test_dataset, args.out_dir)

def reverse_preprocessing(data, dataset):
    """
    Reverse normalization using dataset's mean and std
    """
    device = data.device  # Get the device of `data`
    mean = dataset.global_mean.to(device)
    std = dataset.global_std.to(device)
    
    return (data * std) + mean  # Now all tensors are on the same device

def infer_and_save(test_loader, model, device, dataset, output_dir, template_mesh_path, dataset_type):
    """
    Run inference on the test dataset and save predictions.

    Parameters:
    - test_loader (DataLoader): DataLoader for the test set.
    - model (torch.nn.Module): The trained model for inference.
    - device (torch.device): Device to run inference on.
    - dataset (InMemoryDataset): The dataset to reverse the preprocessing (mean and std).
    - output_dir (str): Directory to save output files.
    - template_mesh_path (str): Path to the template mesh.
    - dataset_type (str): Type of dataset ('CoMA' or 'ThreeDFN').
    """
    model.eval()
    predictions = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)  # Move data to the correct device
            out = model(data.x)    # Forward pass

            pred = out if not isinstance(out, tuple) else out[0]  # Handle tuple case
            if args.dataset == 'ThreeDFN' or args.dataset == 'foundation':
                pred = reverse_preprocessing(pred, dataset)  # Reverse preprocessing
            for i, filename in enumerate(data.filename):  # Use filename attribute
                predictions.append((pred[i], filename))  # Store predictions with filenames
    
    save_predictions(predictions, output_dir, template_mesh_path, dataset_type)

def save_predictions(predictions, output_dir, template_mesh_path, dataset_type):
    """
    Saves the model's predictions as mesh files (.ply or .obj) using input file names.

    Parameters:
    - predictions (list of tuples): Each tuple contains (tensor prediction, input filename).
    - output_dir (str): Directory to save the output files.
    - template_mesh_path (str): Path to the template mesh to extract faces.
    - dataset_type (str): Type of dataset ('CoMA' or 'ThreeDFN').
    """
    if dataset_type == 'ThreeDFN' or dataset_type == 'foundation':
        template_mesh = Mesh(filename=template_mesh_path)  # Load template mesh for ThreeDFN

    for pred, filename in predictions:
        pred_np = pred.detach().cpu().numpy()  # Convert to NumPy array
        pred_np = pred_np.reshape(-1, 3)  # Ensure correct shape

        if dataset_type == 'CoMA':
            output_filename = filename.replace('.ply', '_pred.ply')  # Modify filename
            meshdata.save_mesh(os.path.join(output_dir, output_filename), pred_np)  # Use CoMA's save_mesh
        else:  # ThreeDFN
            output_filename = filename.replace('.obj', '_pred.obj')  # Modify filename
            mesh = Mesh(v=pred_np, f=template_mesh.f)  # Use template faces
            mesh.write_obj(os.path.join(output_dir, output_filename))

output_dir = osp.join(args.data_fp, 'predicted')  # Output directory to save the predictions

# Run inference and save predictions
if args.dataset == 'CoMA':
    infer_and_save(test_loader, model, device, meshdata.test_dataset, output_dir, template_fp, 'CoMA')
elif args.dataset == 'ThreeDFN':
    infer_and_save(test_loader, model, device, test_dataset, output_dir, template_fp, 'ThreeDFN')
elif args.dataset == 'foundation':
    infer_and_save(test_loader, model, device, test_dataset, output_dir, template_fp, 'foundation')

def get_latent_embeddings(test_loader, model, device, data_fp, prefix):
    model.eval()
    path = osp.join(data_fp, f'{prefix}_latent_embeddings.pt')
    latent_embeddings = []
    filenames = []  # Collect filenames corresponding to each latent embedding
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            latent = model.encoder(data.x)  # Extract latent embeddings 
            latent_embeddings.append(latent)
            filenames.extend(data.filename)  # Collect the filenames

    # Convert list to tensor
    latent_embeddings = torch.cat(latent_embeddings, dim=0)
    
    # Save the embeddings and filenames together
    torch.save({
        'embeddings': latent_embeddings,
        'filenames': filenames
    }, path)

    return latent_embeddings, filenames

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

latent_embeddings, f = get_latent_embeddings(test_loader, model, device, args.data_fp, "test")

tsne = TSNE(n_components=2)
latent_2d = tsne.fit_transform(latent_embeddings.cpu().numpy())

plt.scatter(latent_2d[:, 0], latent_2d[:, 1])
plt.title('t-SNE visualization of latent space')
plt.show()

latent_embeddings, f = get_latent_embeddings(test_loader, model, device, args.data_fp, "train")

tsne = TSNE(n_components=2)
latent_2d = tsne.fit_transform(latent_embeddings.cpu().numpy())

plt.scatter(latent_2d[:, 0], latent_2d[:, 1])
plt.title('t-SNE visualization of latent space')
plt.show()