from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_FOLDER = PROJECT_ROOT / "data" / "FaceForensics++_C23"
CSV_FOLDER = DATA_FOLDER / "csv"

OUTPUT_DIR = "output"
MODEL_DIR = f"{OUTPUT_DIR}/models"
