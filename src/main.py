from example.test import say_hello
from data.dataset_builder import DatasetBuilder


if __name__ == "__main__":
    message = say_hello()
    print(message)
    
    
    builder = DatasetBuilder()
    builder.extract_frame()