from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, folder):
        self.dataset = folder

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Returns:
            cube: B, 1, H, W, D
            segmentation: B, 1, H, W, D
        """
        loaded = torch.load(self.dataset[index])
        cube, seg = loaded["data"], loaded["value"]
        cube = cube/255.0
    
        return cube, seg