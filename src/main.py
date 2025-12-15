import logging
from dataset_builder.dataset_builder import DatasetBuilder


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    message = "Hello veridisQuo!"
    print(message)

    builder = DatasetBuilder(mode='test', num_workers=8)
    builder.build_complete_dataset()