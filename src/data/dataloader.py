from dataclasses import dataclass


@dataclass
class FaceItem:
    pass



class DataLoader:
    def __init__(self, batch_size : int = 32 , shuffle : bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        pass