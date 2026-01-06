"""
Example script demonstrating the usage of InferenceEngine for deepfake detection.

This script shows how to:
1. Initialize the inference engine
2. Perform inference on a single image
3. Perform inference on a batch of images
4. Perform inference on a video
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import InferenceEngine, InferenceResult, VideoInferenceResult


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def example_single_image(engine: InferenceEngine, image_path: str):
    """Example: Inference on a single image.

    Parameters:
        engine (InferenceEngine): Initialized inference engine
        image_path (str): Path to image file
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Image Inference")
    print("="*60)

    # Perform inference
    result = engine.predict_image(image_path)

    # Print results
    print(f"\nImage: {image_path}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Probabilities:")
    for label, prob in result.probabilities.items():
        print(f"  {label}: {prob:.4f}")


def example_batch_inference(engine: InferenceEngine, image_paths: list):
    """Example: Inference on a batch of images.

    Parameters:
        engine (InferenceEngine): Initialized inference engine
        image_paths (list): List of paths to image files
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Inference")
    print("="*60)

    # Perform batch inference
    results = engine.predict_batch(image_paths)

    # Print results
    print(f"\nProcessed {len(results)} images:")
    for i, (path, result) in enumerate(zip(image_paths, results), 1):
        print(f"\n{i}. {Path(path).name}")
        print(f"   Prediction: {result.prediction} (confidence: {result.confidence:.4f})")


def example_video_inference(engine: InferenceEngine, video_path: str):
    """Example: Inference on a video.

    Parameters:
        engine (InferenceEngine): Initialized inference engine
        video_path (str): Path to video file
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Video Inference")
    print("="*60)

    # Perform video inference
    result = engine.predict_video(
        video_path,
        frames_per_second=1,
        max_frames=30,
        aggregate_method="majority"
    )

    # Print results
    print(f"\nVideo: {video_path}")
    print(f"Frames analyzed: {result.num_frames_analyzed}")
    print(f"Overall prediction: {result.aggregate_prediction}")
    print(f"Overall confidence: {result.aggregate_confidence:.4f}")
    print(f"\nFrame-by-frame results:")
    for i, frame_result in enumerate(result.frame_results[:10], 1):  # Show first 10
        print(f"  Frame {i}: {frame_result.prediction} ({frame_result.confidence:.4f})")
    if len(result.frame_results) > 10:
        print(f"  ... and {len(result.frame_results) - 10} more frames")


def example_model_info(engine: InferenceEngine):
    """Example: Get model information.

    Parameters:
        engine (InferenceEngine): Initialized inference engine
    """
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)

    info = engine.get_model_info()

    print(f"\nModel path: {info['model_path']}")
    print(f"Device: {info['device']}")
    print(f"Total parameters: {info['num_parameters']:,}")
    print(f"Trainable parameters: {info['num_trainable_parameters']:,}")
    print(f"Use face detection: {info['use_face_detection']}")
    print(f"Input image size: {info['image_size']}")
    print(f"Class labels: {info['class_labels']}")


def main():
    """Main function demonstrating InferenceEngine usage."""
    setup_logging()

    print("="*60)
    print("VeridisQuo Inference Engine - Usage Examples")
    print("="*60)

    # Initialize inference engine
    print("\nInitializing InferenceEngine...")
    engine = InferenceEngine(
        device='cuda',  # Use 'cpu' if CUDA is not available
        auto_download=True,
        use_face_detection=True
    )

    # Show model info
    example_model_info(engine)

    # Example 1: Single image inference
    # Replace with actual image path
    image_path = "path/to/your/image.jpg"
    if Path(image_path).exists():
        example_single_image(engine, image_path)
    else:
        print(f"\nSkipping single image example: {image_path} not found")

    # Example 2: Batch inference
    # Replace with actual image paths
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg"
    ]
    existing_paths = [p for p in image_paths if Path(p).exists()]
    if existing_paths:
        example_batch_inference(engine, existing_paths)
    else:
        print(f"\nSkipping batch inference example: no images found")

    # Example 3: Video inference
    # Replace with actual video path
    video_path = "path/to/your/video.mp4"
    if Path(video_path).exists():
        example_video_inference(engine, video_path)
    else:
        print(f"\nSkipping video inference example: {video_path} not found")

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
