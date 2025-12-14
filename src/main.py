from data.dataset_builder import DatasetBuilder


if __name__ == "__main__":
    message = "Hello veridisQuo!"
    print(message)    
    
    builder = DatasetBuilder(mode='test', num_workers=8)
    builder.extract_frame()
    builder.extract_face()