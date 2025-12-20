import logging
import torch
from dataset_builder.dataset_builder import DatasetBuilder
from training.trainer import Trainer
from dotenv import load_dotenv, find_dotenv
from utils.device import get_device
from core.model import DeepFakeDetector
from torchsummary import summary
import time

load_dotenv(find_dotenv())

if __name__ == "__main__":
    
    
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    device = get_device()
    
    model = DeepFakeDetector().to(device)
    summary(model, (3,224,224))
    

    trainer = Trainer()
    print("strat training loop")
    trainer.train()
    
    
    
    