from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "FaceForensics++_C23"
TEST_DATA_FOLDER = PROJECT_ROOT / "data" / "FaceForensics++_C23_test"
CSV_FOLDER = DATA_FOLDER / "csv"
FRAME_DIR = DATA_FOLDER / "frames"
FACES_DIR = DATA_FOLDER / "faces"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "models"

