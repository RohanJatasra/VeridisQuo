"""
Data schemas for CSV files.
"""

from dataclasses import dataclass


@dataclass
class CSVItem:
    File_Path : str
    Label : str
    Frame_Count : int
    Width : int
    Height : int
    Codec : str
    File_Size_MB : float
     
@dataclass
class FrameCSVItem:
    Frame_Path: str
    Video_Path: str
    Label: str
    Width: int
    Height: int
    Frame_Number: int
    Timestamp: float
    Dataset: str
      
@dataclass
class FaceCSVItem:
    Face_Path: str
    Frame_Path: str
    Video_Path: str
    Label: str
    Frame_Number: int
    Dataset: str
    Confidence: float
    BBox_X: int
    BBox_Y: int
    BBox_Width: int
    BBox_Height: int
    Face_Width: int
    Face_Height: int
    
    
