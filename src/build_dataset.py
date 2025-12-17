import logging
import torch
from dataset_builder.dataset_builder import DatasetBuilder
from training.trainer import Trainer
from dotenv import load_dotenv, find_dotenv
from utils.device import get_device

load_dotenv(find_dotenv())

if __name__ == "__main__":
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    device = get_device()

    # builder = DatasetBuilder(mode='test', num_workers=8, device=device)
    # builder.build_complete_dataset(name="test_preprocessed_dataset")
    
    builder = DatasetBuilder(mode='train', num_workers=8, device=device)
    builder.build_complete_dataset(name="preprocessed_dataset")
    