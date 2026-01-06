from logging import basicConfig, StreamHandler, FileHandler, getLogger
import argparse
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from inference import InferenceEngine
from utils import ConfigService
from dataset_builder import DatasetBuilder
from training import Trainer

SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def retrieve_args() -> argparse.Namespace:
    """Retrieve command-line arguments.
    Returns:
        argparse.Namespace: parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description='VeridisQuo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='path to configuration file (JSON or YAML)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'inference', 'build_dataset'],
        default='inference',
        help='execution mode'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='input file or directory for inference'
    )
    return parser.parse_args()
    
def init_logging_system(
    log_dir: str,
    log_filename: str, 
    log_level: int,
    log_format: str,
    log_datefmt: str,
    console: bool =True,
    file: bool =True
):
    #create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    handlers = []
    if console:
        handlers.append(StreamHandler())
    if file:
        handlers.append(FileHandler(log_filename, mode='w', encoding='utf-8'))
        
    basicConfig(
        filename=log_filename, 
        level=log_level, 
        format=log_format,
        datefmt=log_datefmt,
        handlers=handlers
    )

def check_input_file(input_path: Path) -> bool:
    """Check if the input file is a video or an image based on its extension.
    Parameters:
        input_path (Path): path to the input file
    Returns:
        bool: True if video, False otherwise
    Raises:
        ValueError: if file extension is unsupported
    """
    assert input_path.is_file(), f"Input path is not a file: {input_path}"
    assert input_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS.extend(SUPPORTED_IMAGE_EXTENSIONS), \
        "Unsupported file extension: {input_path.suffix}. Supported extensions: " + \
        f"Videos: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}; Images: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
        
    return input_path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        

if __name__ == "__main__":
    #load environment variables
    load_dotenv(find_dotenv())

    #retrieve command-line arguments
    args = retrieve_args()
    try:
        assert args.config, "No configuration file provided. Use --config to specify the config file."
        assert Path(args.config).exists(), f"Configuration file not found: {args.config}"
    
        config: ConfigService = ConfigService(ConfigService.load_config(args.config))
        config_type = config.get_config_type()
        
        assert config_type in args.mode, "Configuration type does not match execution mode."
    
    except AssertionError as e:
        print(f"Could not start VeridisQuo, invalid parameter: {e}")
        exit(1)

    try:
        logger_params = config.extract_logger_config()
        init_logging_system(**logger_params)

        logger = getLogger("/".join(__file__.split("/")[-2:]))
        logger.info(f"Logging initialized to file: {logger_params.get('log_filename', 'N/A')}")
        
        #execution based on mode
        match args.mode:
            case 'inference':
                logger.info("Starting inference mode...")

                assert args.input, "Input path is required for inference mode."
                
                engine = InferenceEngine(config=config.extract_inference_config())

                input_path = Path(args.input)
                assert input_path.exists(), f"Input path not found: {input_path}"

                              
                
                if input_path.is_file():
                    logger.info(f"Running inference on file: {input_path}")
                    check_input_file(input_path)
                    
                    result = engine.predict_video(
                        video_paths=input_path,
                        **config.extract_predict_video_params()
                    )
                    
                elif input_path.is_dir():
                    input_files = list(input_path.glob('*'))
                    logger.info(f"Running inference on directory: {input_path} ({len(input_files)} files found)")
                    
                    for file in input_files:
                        check_input_file(file)
                    
                    result = engine.predict_video_batch(
                        video_paths=input_files,
                        **config.extract_predict_video_params()
                    )

            case 'train':
                logger.info("Starting training mode...")
                logger.warning("Training mode is under development and may not work as expected.")
                
                trainer = Trainer(config=config.extract_training_config()) 
                trainer.train()

            case 'build_dataset':
                logger.info("Starting dataset building mode...")
                logger.warning("Dataset building mode is experimental and may not work as expected.")

                builder = DatasetBuilder(**config.extract_dataloader_config())
                builder.build_complete_dataset(**config.extract_build_dataset_params())

        logger.info("Execution completed successfully")
        
    except KeyError as e:
        err_mess = f"Configuration key error: {e}"
    
    except AssertionError as e:
        err_mess = f"Invalid data: {e}"
    
    except Exception as e:
        err_mess = f"An unexpected error occurred: {e}"
        
    finally:
        logger.fatal(err_mess)
        print(err_mess)
        exit(1)