import torch
from torch.utils.data import Dataset


class NinaPro_Dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).type(dtype=torch.FloatTensor)
        self.y = torch.from_numpy(y).type(dtype=torch.FloatTensor)
        self.len = self.x.shape[0]

    def __getitem__(self, item):
        x_i = self.x[item, :]
        y_i = self.y[item]
        return x_i, y_i

    def __len__(self):
        return self.len
