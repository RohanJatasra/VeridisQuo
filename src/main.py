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
    
    # trainer = Trainer(device=device)
    # trainer.train()
    
    model = DeepFakeDetector().to(device)
    summary(model, (3,224,224))
    
    fake_batch = torch.randn(8, 3, 224, 224).to(device)
    
    start = time.time()
    output = model(fake_batch)
    end = time.time()
    
    print(f"Output shape: {output.shape}")
    print(f"time : {end - start}")
    
    trainer = Trainer()
    
    
    train_dataloader = trainer.train_loader
    
    for batch_idx, (data, target) in enumerate(train_dataloader):
        print(f"Batch {batch_idx}: data shape = {data.shape}, target shape = {target.shape}")
        
    print("strat training loop")
    trainer.train()
    
    
    
    