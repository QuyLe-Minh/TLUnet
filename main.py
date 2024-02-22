from data_reader.ircadb import *
from data_reader.lits import *
from data_reader.sliver import *
from data_loader import *
from torch.utils.data import DataLoader
from train import training
from utils import config
from eval import eval
    

if __name__ == "__main__":
    sample_dataset, sample_val = 0, 0
    sample_dataset, sample_val = read_ircadb("dataset/ircadb", sample_dataset, sample_val)
    sample_dataset, sample_val = read_sliver("dataset/sliver", sample_dataset, sample_val)
    sample_dataset, sample_val = read_lits("dataset/LITS17", sample_dataset, sample_val)
    
    train_dataset = CustomDataset(config.train)
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle = True, num_workers=2)

    val_dataset = CustomDataset(config.val)    
    val_loader = DataLoader(val_dataset, 1, shuffle=False, num_workers=2)

    training(config, train_loader, val_loader, config.mode)
    eval(config, val_loader, "tlu.pt")