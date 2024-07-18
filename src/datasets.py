import os

import torch


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()  # Calling the constructor of the parent class
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"  # Checking if split is valid
        self.split = split  # Setting the split type
        self.num_classes = 1854  # Setting the number of classes
        
        # Loading data and subject indices based on the split type
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))  # Loading labels for train/val sets
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."  # Checking if number of unique labels matches the number of classes

        # Calculate mean and standard deviation
        # mean = torch.mean(self.X)
        # std = torch.std(self.X)
        # mean = -0.006601160857826471
        # std = 2.5331075191497803
        
        # Standardization
        # self.X = (self.X - mean) / std
        # Standardize in-place
        # self.X.sub_(mean).div_(std)

    def __len__(self) -> int:
        return len(self.X)  # Returning the length of the dataset
    
    def __getitem__(self, i):
        if hasattr(self, "y"):  # Checking if labels are available
            return self.X[i], self.y[i], self.subject_idxs[i]  # Returning data, label, and subject index
        else:
            return self.X[i], self.subject_idxs[i]  # Returning data and subject index
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]  # Returning the number of channels in the data
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]  # Returning the sequence length of the data
