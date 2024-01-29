from data_reader.ircadb import *
from data_reader.lits import *
from data_loader import *
from torch.utils.data import DataLoader
from train import training
from utils import Config
    

if __name__ == "__main__":
    if not os.path.exists("dataset/dataset.pth"):
        read_ircadb("dataset/ircadb")
        read_lits("dataset/LITS17")
    
    config = Config()
    train_dataset = CustomDataset(config.train)
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle = True, num_workers=2)
    
    val_dataset = CustomDataset(config.val)    
    val_loader = DataLoader(val_dataset, config.batch_size, shuffle=False, num_workers=2)
    
    training(config, train_loader, val_loader, config.mode)