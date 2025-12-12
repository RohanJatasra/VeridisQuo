import os
import csv
from dataclasses import dataclass


DATA_FOLDER = "/data/FaceForensics++_C23/"
CSV_FOLDER = os.path.join(DATA_FOLDER, "csv")


@dataclass
class CSVFile:
    File_Path : str
    Label : str
    Frame_Count : int
    Width : int
    Height : int
    Codec : str
    File_Size_MB : float
    
    


class CustomDataset:
    def __init__(self):
        pass
    
    
    def load_all_csvs(self) -> dict[str, list[CSVFile]]:
        csv_files = os.listdir(CSV_FOLDER)
        data = {}
        for csv_file in csv_files:
            if csv_file.endswith('.csv'):
                data[csv_file] = self.load_csv(csv_file)
        return data
    
    
    def load_csv(self, csv_filename) -> list[CSVFile]:
        csv_path = os.path.join(CSV_FOLDER, csv_filename)
        data = []
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                csv_file = CSVFile(
                    File_Path=row.get("File Path", ""),
                    Label=row.get("Label", ""),
                    Frame_Count=int(row.get("Frame Count", 0)),
                    Width=int(row.get("Width", 0)),
                    Height=int(row.get("Height", 0)),
                    Codec=row.get("Codec", ""),
                    File_Size_MB=float(row.get("File Size(MB)", 0.0))
                )
                data.append(csv_file)
        return data