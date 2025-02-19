import os
import os.path as osp
import shutil
import torch
from torch_geometric.data import InMemoryDataset, Data
import trimesh

def read_mesh(file_path):
    mesh = trimesh.load_mesh(file_path)
    data = Data()
    data.x = torch.tensor(mesh.vertices, dtype=torch.float)
    data.pos = data.x.clone()
    data.face = torch.tensor(mesh.faces, dtype=torch.long)
    return data

class ThreeDFN(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None, normalize=True):
        self.normalize = normalize  
        super(ThreeDFN, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path, weights_only=False)
        
        # Store global mean and std after loading processed data
        self.global_mean = torch.load(osp.join(root, 'global_mean.pt'))
        self.global_std = torch.load(osp.join(root, 'global_std.pt'))

    @property
    def raw_file_names(self):   
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.obj')]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    @property
    def mean(self):
        # After processing, self.data.x is already normalized if normalization was applied.
        # Here we return the global mean computed from the processed data.
        return self.data.x.mean(dim=0)

    @property
    def std(self):
        return self.data.x.std(dim=0)

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please manually place the .obj files in {}'.format(self.raw_dir)
        )

    def process(self):
        obj_files = sorted(self.raw_file_names)
        data_list = []

        for file in obj_files:
            file_path = osp.join(self.raw_dir, file)
            data = read_mesh(file_path)
            data.filename = file
            data_list.append(data)

        all_x = torch.cat([data.x for data in data_list], dim=0)
        global_mean = all_x.mean(dim=0)
        global_std = all_x.std(dim=0)

        # Save for later retrieval
        torch.save(global_mean, osp.join(self.root, 'global_mean.pt'))
        torch.save(global_std, osp.join(self.root, 'global_std.pt'))

        if self.normalize:
            for data in data_list:
                data.x = (data.x - global_mean) / global_std

        split_idx = int(0.8 * len(data_list))
        torch.save(self.collate(data_list[:split_idx]), self.processed_paths[0])
        torch.save(self.collate(data_list[split_idx:]), self.processed_paths[1])