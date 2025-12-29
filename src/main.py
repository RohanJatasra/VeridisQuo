from logging import basicConfig, StreamHandler, FileHandler, getLogger, info
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
from config import LOGS_DIR, LOG_FILE, LOG_LEVEL, LOG_FORMAT, LOG_DATEFMT
from inference import InferenceEngine

def init_logging_system():
    #create logs directory if it doesn't exist
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    basicConfig(
        filename=LOG_FILE, 
        level=LOG_LEVEL, 
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=[
            StreamHandler(),  #console output
            FileHandler(LOG_FILE, mode='w', encoding='utf-8')  #file output
        ]
    )

if __name__ == "__main__":
    load_dotenv(find_dotenv())

    init_logging_system()
    logger = getLogger("/".join(__file__.split("/")[-2:]))

    #log the file location
    info(f"Logging initialized to file: {LOG_FILE.resolve()}")

    # InferenceEngine()

    # device = get_device()
    
    # model = DeepFakeDetector().to(device)
    # summary(model, (3,224,224))
    

    # trainer = Trainer()
    # print("strat training loop")
    # trainer.train()
    
    
    
    