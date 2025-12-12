import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepfake-detector'))

from data.dataset import CustomDataset, CSVFile

class TestCustomDataset:
    pass
    # def test_load_csv(self):
    #     dataset = CustomDataset()
    #     csv_data = dataset.load_csv("original.csv")
        
    #     assert isinstance(csv_data, list), "CSV data should be a list"
    #     assert all(isinstance(item, CSVFile) for item in csv_data), "All items should be instances of CSVFile"
        
    # def test_load_all_csvs(self):
    #     dataset = CustomDataset()
    #     all_csv_data = dataset.load_all_csvs()
        
    #     assert isinstance(all_csv_data, dict), "All CSV data should be a dictionary"
    #     for key, value in all_csv_data.items():
    #         assert isinstance(key, str), "Dictionary keys should be strings (CSV filenames)"
    #         assert isinstance(value, list), "Dictionary values should be lists of CSVFile instances"
    #         assert all(isinstance(item, CSVFile) for item in value), "All items in the lists should be instances of CSVFile"