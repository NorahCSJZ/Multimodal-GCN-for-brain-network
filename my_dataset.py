import torch
from torch.utils.data import Dataset

# Datasets class ( further implenmented)
class MyDatasets(Dataset):
    def __init__(self, adj, node_features, labels):
        self.x_data = adj
        self.y_data = node_features
        self.z_data = labels

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item], self.z_data[item]

    def __len__(self):
        return len(self.z_data)
