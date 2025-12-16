import logging
import torch
from dataset_builder.dataset_builder import DatasetBuilder


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Device configuration - use GPU if available
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ Using GPU: {device_name}")
        logging.info(f"Using GPU: {device_name}")
    else:
        device = "cpu"
        print("⚠ GPU not available, using CPU")
        logging.info("GPU not available, using CPU")

    message = "Hello veridisQuo!"
    print(message)

    builder = DatasetBuilder(mode='test', num_workers=8, device=device)
    builder.build_complete_dataset(name="test_preprocessed_dataset")