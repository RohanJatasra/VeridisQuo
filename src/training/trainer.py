from pathlib import Path
from data.dataset import Dataset
from data.dataloader import DataLoader
from config import PROJECT_ROOT

import torch


class Trainer:
    def __init__(self, device : torch.device):
        
        self.device = device

        train_dataset = Dataset(file_path=str(PROJECT_ROOT / "data/test_preprocessed_dataset/train.csv"))
        test_dataset = Dataset(file_path=str(PROJECT_ROOT / "data/test_preprocessed_dataset/test.csv"))
        eval_dataset = Dataset(file_path=str(PROJECT_ROOT / "data/test_preprocessed_dataset/eval.csv"))
        
        
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, device=self.device)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, device=self.device)
        self.eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, device=self.device)
        
        print(f"len train_loader: {len(self.train_loader)}")
        print(f"len test_loader: {len(self.test_loader)}")
        print(f"len eval_loader: {len(self.eval_loader)}")
        
        
        
    def train(self):
        pass