from data.dataset import Dataset

import torch

class DataLoader:
    def __init__(self,dataset : Dataset, batch_size : int = 32 , shuffle : bool = True, device : torch.device = torch.device("cpu")):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.current_index = 0
        
        if self.shuffle:
            from random import shuffle
            self.indices = list(range(len(self.dataset)))
            shuffle(self.indices)
            
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
            
    def __iter__(self):
        self.current_index = 0
        return self
    
    def __next__(self):
        if self.current_index >= len(self.dataset):
            raise StopIteration
        
        batch_indices = range(self.current_index, min(self.current_index + self.batch_size, len(self.dataset)))
        if self.shuffle:
            batch = [self.dataset[self.indices[i]] for i in batch_indices]
        else:
            batch = [self.dataset[i] for i in batch_indices]
        
        self.current_index += self.batch_size
        return batch
        
        
    