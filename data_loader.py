from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)//2

    def __getitem__(self, index):
        """
        Returns:
            cube: B, 1, H, W, D
            segmentation: B, 1, H, W, D
        """
        cube = self.dataset[f"data_{index}"]
        segmentation = self.dataset[f"value_{index}"]
    
        return cube, segmentation
