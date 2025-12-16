from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Train dataset folders
DATA_FOLDER = PROJECT_ROOT / "data" / "FaceForensics++_C23"
CSV_FOLDER = DATA_FOLDER / "csv"
FRAME_DIR = DATA_FOLDER / "frames"
FACES_DIR = DATA_FOLDER / "faces"

# Test dataset folders
TEST_DATA_FOLDER = PROJECT_ROOT / "data" / "FaceForensics++_C23_test"
TEST_CSV_FOLDER = TEST_DATA_FOLDER / "csv"
TEST_FRAME_DIR = TEST_DATA_FOLDER / "frames"
TEST_FACES_DIR = TEST_DATA_FOLDER / "faces"

# Output folders
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "models"

# Face detection model
FACE_DETECTION_MODEL_PATH = PROJECT_ROOT / "models" / "face_detection" / "face_detection.pt"

# Preprocessed dataset folders (balanced and split)
PREPROCESSED_DATASET_DIR = PROJECT_ROOT / "data" 

