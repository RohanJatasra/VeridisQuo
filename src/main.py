import logging
import torch
from dataset_builder.dataset_builder import DatasetBuilder
from training.trainer import Trainer
from dotenv import load_dotenv, find_dotenv
from utils.device import get_device
from core.model import DeepFakeDetector
from torchsummary import summary
import time
from datetime import datetime
from pathlib import Path
from config import LOGS_DIR

load_dotenv(find_dotenv())

if __name__ == "__main__":

    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate unique log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGS_DIR / f"training_{timestamp}.log"

    # Configure logging to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file, mode='w', encoding='utf-8')  # File output
        ]
    )

    # Log the file location
    logging.info(f"Logging to file: {log_file}")

    device = get_device()
    
    model = DeepFakeDetector().to(device)
    summary(model, (3,224,224))
    

    trainer = Trainer()
    print("strat training loop")
    trainer.train()
    
    
    
    