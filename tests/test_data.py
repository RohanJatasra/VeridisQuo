import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepfake-detector'))

from data.dataset import CustomDataset, CSVFile
from config import TEST_CSV_FOLDER

class TestCustomDataset:
    def test_load_csv(self):
        """Test loading a single CSV file from test dataset"""
        dataset = CustomDataset(csv_folder=TEST_CSV_FOLDER)

        # Get the first CSV file in the test folder
        csv_files = [f for f in os.listdir(TEST_CSV_FOLDER) if f.endswith('.csv')]
        assert len(csv_files) > 0, "No CSV files found in test dataset"

        csv_data = dataset.load_csv(csv_files[0])

        assert isinstance(csv_data, list), "CSV data should be a list"
        assert all(isinstance(item, CSVFile) for item in csv_data), "All items should be instances of CSVFile"

    def test_load_all_csvs(self):
        """Test loading all CSV files from test dataset"""
        dataset = CustomDataset(csv_folder=TEST_CSV_FOLDER)
        all_csv_data = dataset.load_all_csvs()

        assert isinstance(all_csv_data, dict), "All CSV data should be a dictionary"
        assert len(all_csv_data) > 0, "Should load at least one CSV file"

        for key, value in all_csv_data.items():
            assert isinstance(key, str), "Dictionary keys should be strings (CSV filenames)"
            assert key.endswith('.csv'), "Dictionary keys should be CSV filenames"
            assert isinstance(value, list), "Dictionary values should be lists of CSVFile instances"
            assert all(isinstance(item, CSVFile) for item in value), "All items in the lists should be instances of CSVFile"