"""
Model initialization script for DeepFakeDetector.

This script provides utilities to initialize and load trained models.
Can be used as a standalone script or imported as a module.
"""

import logging
from pathlib import Path
from typing import Optional, Union
import torch
from torchsummary import summary

from utils.load_model import load_model
from utils.device import get_device
from config import PROJECT_ROOT, VERIDISQUO_MODEL_PATH


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)





def main():
    device = get_device()
    model = load_model(VERIDISQUO_MODEL_PATH, device=device, eval_mode=True)
    model.to(device)
    logger.info("Model loaded successfully.")
    
    random_input = torch.randn(1, 3, 224, 224).to(device)
    
    output = model(random_input)
    logger.info(f"Model output shape: {output.shape}")


if __name__ == "__main__":
    main()
