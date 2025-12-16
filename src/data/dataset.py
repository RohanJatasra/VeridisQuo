from core.schemas import CSVItem
from utils.csv_services import CSVService


class Dataset:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.items : list[CSVItem] = CSVService.load_csv(self.file_path, CSVItem)
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index: int) -> CSVItem:
        return self.items[index]
    
    def get_all_items(self) -> list[CSVItem]:
        return self.items
    
    
        