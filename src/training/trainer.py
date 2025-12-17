from data.dataset import Dataset
from data.dataloader import DataLoader
from config import PROJECT_ROOT
import wandb

from dataclasses import dataclass

import torch
import logging




@dataclass
class TrainingConfig:
    device : torch.device = torch.device("cpu")
    batch_size : int = 32
    learning_rate : float = 0.001
    num_epochs : int = 10
    log_interval : int = 10
    save_model_path : str = str(PROJECT_ROOT / "output" / "models" / "model.pth")


class Trainer:
    def __init__(self, config: TrainingConfig):
        
        
        self.logger = logging.getLogger("Trainer")
        
        self.device = config.device
        self.wandb_run = None
        self._init_wandb()

        train_dataset = Dataset(file_path=str(PROJECT_ROOT / "data/test_preprocessed_dataset/train.csv"))
        test_dataset = Dataset(file_path=str(PROJECT_ROOT / "data/test_preprocessed_dataset/test.csv"))
        eval_dataset = Dataset(file_path=str(PROJECT_ROOT / "data/test_preprocessed_dataset/eval.csv"))
        
        
        self.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, device=self.device)
        self.test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, device=self.device)
        self.eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, device=self.device)
        
        
    def _init_wandb(self):
        self.wandb_run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="my-awesome-team-name",
        # Set the wandb project where this run will be logged.
        project="my-awesome-project",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
)
        
        
    def train(self):
        # self.wandb_run.log({"acc": acc, "loss": loss})
        # self.wandb_run.finish()
        pass
    
    
    def _train_step_batch(self):
        pass
    
    def _test_step_batch(self):
        pass
    
    def _eval_model(self):
        pass
    
    def _save_model(self):
        pass
    
    
    
    
    
    