from inference import InferenceEngine
from logging import basicConfig, StreamHandler, FileHandler, getLogger
from dotenv import load_dotenv, find_dotenv
from config import LOGS_DIR, LOG_FILE, LOG_LEVEL, LOG_FORMAT, LOG_DATEFMT, TEST_DATA_FOLDER, PROJECT_ROOT
from utils.config import load_config
from utils.config_dataclasses import InferenceConfig

def init_logging_system():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=[
            StreamHandler(),
            FileHandler(LOG_FILE, mode='w', encoding='utf-8')
        ]
    )

if __name__ == "__main__":
    load_dotenv(find_dotenv())

    init_logging_system()
    logger = getLogger("/".join(__file__.split("/")[-2:]))

    # Load configuration from JSON
    config_path = PROJECT_ROOT / "config.json"
    config_dict = load_config(config_path)
    config = InferenceConfig.from_dict(config_dict)

    # Initialize engine
    engine = InferenceEngine(config=config)


    # Video path
    video_path = TEST_DATA_FOLDER / "DeepFakeDetection" / "01_02__meeting_serious__YVGY8LOK.mp4"

    logger.info(f"Processing video: {video_path}")

    # Run inference
    result = engine.predict_video(
        video_path,
    )

    # Display results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}\n")

    print(f"Video: {result.video_path}")
    print(f"Frames analyzed: {result.num_frames_analyzed}")
    print(f"\nPREDICTION: {result.aggregate_prediction}")
    print(f"CONFIDENCE: {result.aggregate_confidence:.2%}")

    if result.aggregation_metadata:
        print(f"\nAggregation metadata:")
        for key, value in result.aggregation_metadata.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    print(f"\n{'='*60}\n")
