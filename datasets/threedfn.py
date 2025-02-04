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
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        super(ThreeDFN, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.obj')]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    @property
    def mean(self):
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

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        split_idx = int(0.8 * len(data_list))
        torch.save(self.collate(data_list[:split_idx]), self.processed_paths[0])
        torch.save(self.collate(data_list[split_idx:]), self.processed_paths[1])
    
        # # Cleanup raw data after processing
        # shutil.rmtree(self.raw_dir)