#!/usr/bin/env python3
"""
Command-line interface for VeridisQuo deepfake detection inference.

This script provides a simple CLI for running deepfake detection on images and videos.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import InferenceEngine


def setup_logging(verbose: bool = False):
    """Setup logging configuration.

    Parameters:
        verbose (bool): Enable verbose (DEBUG) logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def predict_image_cli(args):
    """Run inference on image(s).

    Parameters:
        args: Parsed command-line arguments
    """
    # Initialize engine
    engine = InferenceEngine(
        device=args.device,
        auto_download=args.auto_download,
        use_face_detection=args.use_face_detection
    )

    # Get image paths
    image_paths = [str(p) for p in args.input]

    if len(image_paths) == 1:
        # Single image
        result = engine.predict_image(image_paths[0])

        print("\n" + "="*60)
        print(f"Image: {image_paths[0]}")
        print("="*60)
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.4f}")
        print("\nProbabilities:")
        for label, prob in result.probabilities.items():
            print(f"  {label}: {prob:.4f}")
        print("="*60 + "\n")

    else:
        # Batch of images
        results = engine.predict_batch(image_paths)

        print("\n" + "="*60)
        print(f"Batch Inference Results ({len(results)} images)")
        print("="*60)

        for path, result in zip(image_paths, results):
            print(f"\n{Path(path).name}")
            print(f"  Prediction: {result.prediction}")
            print(f"  Confidence: {result.confidence:.4f}")

        print("\n" + "="*60 + "\n")


def predict_video_cli(args):
    """Run inference on video.

    Parameters:
        args: Parsed command-line arguments
    """
    # Initialize engine
    engine = InferenceEngine(
        device=args.device,
        auto_download=args.auto_download,
        use_face_detection=args.use_face_detection
    )

    # Run video inference
    result = engine.predict_video(
        args.input,
        frames_per_second=args.fps,
        max_frames=args.max_frames,
        aggregate_method=args.aggregate
    )

    print("\n" + "="*60)
    print(f"Video: {args.input}")
    print("="*60)
    print(f"\nOverall Prediction: {result.aggregate_prediction}")
    print(f"Overall Confidence: {result.aggregate_confidence:.4f}")
    print(f"Frames Analyzed: {result.num_frames_analyzed}")

    if args.show_frames:
        print("\nFrame-by-frame results:")
        for i, frame_result in enumerate(result.frame_results, 1):
            print(f"  Frame {i:3d}: {frame_result.prediction} ({frame_result.confidence:.4f})")

    print("="*60 + "\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VeridisQuo Deepfake Detection - Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict single image
  python inference_cli.py image path/to/image.jpg

  # Predict multiple images
  python inference_cli.py image img1.jpg img2.jpg img3.jpg

  # Predict video
  python inference_cli.py video path/to/video.mp4 --fps 1 --max-frames 30

  # Use CPU instead of GPU
  python inference_cli.py image path/to/image.jpg --device cpu

  # Show frame-by-frame results for video
  python inference_cli.py video video.mp4 --show-frames
        """
    )

    # Global arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference (default: cuda)'
    )
    parser.add_argument(
        '--auto-download',
        action='store_true',
        default=True,
        help='Automatically download model from Hugging Face if not found (default: True)'
    )
    parser.add_argument(
        '--no-face-detection',
        dest='use_face_detection',
        action='store_false',
        default=True,
        help='Disable face detection for video inference'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', required=True, help='Command to run')

    # Image inference subcommand
    image_parser = subparsers.add_parser('image', help='Run inference on image(s)')
    image_parser.add_argument(
        'input',
        type=Path,
        nargs='+',
        help='Path to image file(s)'
    )

    # Video inference subcommand
    video_parser = subparsers.add_parser('video', help='Run inference on video')
    video_parser.add_argument(
        'input',
        type=str,
        help='Path to video file'
    )
    video_parser.add_argument(
        '--fps',
        type=int,
        default=1,
        help='Frames per second to extract (default: 1)'
    )
    video_parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to analyze (default: all)'
    )
    video_parser.add_argument(
        '--aggregate',
        type=str,
        default='majority',
        choices=['majority', 'average'],
        help='Aggregation method for frame predictions (default: majority)'
    )
    video_parser.add_argument(
        '--show-frames',
        action='store_true',
        help='Show frame-by-frame results'
    )

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Run appropriate command
    try:
        if args.command == 'image':
            predict_image_cli(args)
        elif args.command == 'video':
            predict_video_cli(args)
    except Exception as e:
        logging.error(f"Error during inference: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == '__main__':
    main()
