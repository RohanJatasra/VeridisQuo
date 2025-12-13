"""
DatasetBuilder module for extracting frames and faces from videos.

This module provides functionality to:
- Extract frames from videos in parallel
- Detect and extract faces from frames in parallel
- Generate CSV metadata files tracking all extracted data
"""

from __future__ import annotations

# Suppress TensorFlow verbose output BEFORE any TF imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import csv
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

import cv2
from tqdm import tqdm

from preprocessing.FramesExtractor import FramesExtractor
from preprocessing.FaceDetector import FaceDetector
from preprocessing.FaceExtractor import FaceExtractor
from data.dataset import CustomDataset, CSVFile
from config import (
    FACES_DIR, FRAME_DIR, DATA_FOLDER, CSV_FOLDER,
    TEST_FACES_DIR, TEST_FRAME_DIR, TEST_DATA_FOLDER, TEST_CSV_FOLDER
)


@dataclass
class ProcessingConfig:
    """Configuration for face detection and extraction."""

    # Frame extraction
    frames_per_second: int = 1

    # Face detection
    min_face_size: int = 40
    confidence_threshold: float = 0.7
    only_keep_top_face: bool = True

    # Face extraction
    face_target_size: Tuple[int, int] = (224, 224)
    face_normalization: str = "zero_one"
    face_padding: float = 0.5


class DatasetBuilder:
    """
    Build datasets by extracting frames and faces from videos in parallel.

    Attributes:
        mode: Operating mode ('train' or 'test')
        num_workers: Number of parallel workers for multiprocessing
        config: Processing configuration
    """

    # CSV field names
    FRAME_CSV_FIELDS = [
        "Frame_Path", "Video_Path", "Label", "Width", "Height",
        "Frame_Number", "Timestamp", "Dataset"
    ]

    FACE_CSV_FIELDS = [
        "Face_Path", "Frame_Path", "Video_Path", "Label",
        "Frame_Number", "Dataset", "Confidence",
        "BBox_X", "BBox_Y", "BBox_Width", "BBox_Height",
        "Face_Width", "Face_Height"
    ]

    # Metadata file names
    FRAMES_METADATA_FILE = "frames_metadata.csv"
    FACES_METADATA_FILE = "faces_metadata.csv"

    def __init__(
        self,
        mode: str = 'train',
        num_workers: Optional[int] = None,
        config: Optional[ProcessingConfig] = None
    ):
        """
        Initialize DatasetBuilder.

        Args:
            mode: Dataset mode - 'train' or 'test'
            num_workers: Number of parallel workers. Defaults to CPU count.
            config: Processing configuration. Defaults to ProcessingConfig().

        Raises:
            ValueError: If mode is not 'train' or 'test'
        """
        if mode not in ['train', 'test']:
            raise ValueError(f"Mode must be 'train' or 'test', got '{mode}'")

        self.mode = mode
        self.num_workers = num_workers if num_workers is not None else cpu_count()
        self.config = config or ProcessingConfig()

        # Set directories based on mode
        self._setup_directories()

        # Initialize dataset and logger
        self.dataset = CustomDataset(csv_folder=self.csv_folder)
        self.videos_path: Dict[str, List[CSVFile]] = self.dataset.load_all_csvs()
        self.logger = logging.getLogger(f"DatasetBuilder[{mode}]")

        self.logger.info(
            f"Initialized DatasetBuilder in {mode} mode with {self.num_workers} workers"
        )

    def _setup_directories(self) -> None:
        """Configure directory paths based on operating mode."""
        if self.mode == 'train':
            self.csv_folder = CSV_FOLDER
            self.data_folder = DATA_FOLDER
            self.frame_dir = FRAME_DIR
            self.faces_dir = FACES_DIR
        else:  # test mode
            self.csv_folder = TEST_CSV_FOLDER
            self.data_folder = TEST_DATA_FOLDER
            self.frame_dir = TEST_FRAME_DIR
            self.faces_dir = TEST_FACES_DIR

    # ==================== Frame Extraction ====================

    def extract_frame(self) -> List[Dict[str, Any]]:
        """
        Extract frames from all videos in parallel.

        Returns:
            List of frame metadata dictionaries
        """
        os.makedirs(self.frame_dir, exist_ok=True)

        # Prepare video list
        all_videos = self._collect_videos()

        self.logger.info(
            f"Starting frame extraction for {len(all_videos)} videos "
            f"using {self.num_workers} workers..."
        )

        # Process in parallel
        csv_data = self._parallel_process_videos(all_videos)

        # Save metadata
        self._save_frames_metadata(csv_data)

        self.logger.info(
            f"✓ Frame extraction complete! "
            f"{len(csv_data)} frames extracted from {len(all_videos)} videos"
        )

        return csv_data

    def _collect_videos(self) -> List[Tuple[str, CSVFile]]:
        """Collect all videos from loaded CSV files."""
        videos = []
        for csv_name, video_list in self.videos_path.items():
            for video in video_list:
                videos.append((csv_name, video))
        return videos

    def _parallel_process_videos(
        self,
        videos: List[Tuple[str, CSVFile]]
    ) -> List[Dict[str, Any]]:
        """Process videos in parallel to extract frames."""
        process_func = partial(
            self._process_single_video,
            data_folder=str(self.data_folder),
            frame_dir=str(self.frame_dir),
            fps=self.config.frames_per_second
        )

        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, videos),
                total=len(videos),
                desc="Extracting frames",
                unit="video"
            ))

        # Flatten results
        return [frame for video_frames in results for frame in video_frames]

    @staticmethod
    def _process_single_video(
        video_data: Tuple[str, CSVFile],
        data_folder: str,
        frame_dir: str,
        fps: int
    ) -> List[Dict[str, Any]]:
        """
        Worker function to extract frames from a single video.

        Args:
            video_data: Tuple of (csv_name, video CSVFile object)
            data_folder: Base data folder path
            frame_dir: Directory to save extracted frames
            fps: Frames per second to extract

        Returns:
            List of frame metadata dictionaries
        """
        csv_name, video = video_data
        video_path = video.File_Path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        complete_video_path = os.path.join(data_folder, video_path)

        if not os.path.exists(complete_video_path):
            return []

        try:
            extractor = FramesExtractor(complete_video_path, nb_fps=fps)
            output_dir = os.path.join(frame_dir, video_name)
            os.makedirs(output_dir, exist_ok=True)

            frames_data = []
            for frame, metadata in extractor.extract_frames():
                frame_filename = f"frame_{metadata['frame_number']:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)

                extractor.save_frame(frame, frame_path)

                frames_data.append({
                    "Frame_Path": frame_path,
                    "Video_Path": video_path,
                    "Label": video.Label,
                    "Width": extractor.width,
                    "Height": extractor.height,
                    "Frame_Number": metadata['frame_number'],
                    "Timestamp": metadata['timestamp'],
                    "Dataset": csv_name.replace('.csv', '')
                })

            return frames_data

        except Exception as e:
            logger = logging.getLogger("DatasetBuilder.Worker")
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return []

    def _save_frames_metadata(self, csv_data: List[Dict[str, Any]]) -> None:
        """Save frame extraction metadata to CSV file."""
        metadata_path = os.path.join(self.frame_dir, self.FRAMES_METADATA_FILE)

        self.logger.info(f"Saving metadata to {metadata_path}...")

        with open(metadata_path, 'w', newline='') as csvfile:
            if csv_data:
                writer = csv.DictWriter(csvfile, fieldnames=self.FRAME_CSV_FIELDS)
                writer.writeheader()
                writer.writerows(csv_data)

        self.logger.info("✓ Metadata saved")

    # ==================== Face Extraction ====================

    def extract_face(self) -> List[Dict[str, Any]]:
        """
        Extract faces from frames in parallel.

        Returns:
            List of face metadata dictionaries

        Raises:
            FileNotFoundError: If frames metadata CSV doesn't exist
        """
        os.makedirs(self.faces_dir, exist_ok=True)

        # Load frames metadata
        frames_data = self._load_frames_metadata()

        self.logger.info(
            f"Processing {len(frames_data)} frames for face extraction "
            f"using {self.num_workers} workers..."
        )

        # Process in parallel
        faces_csv_data, stats = self._parallel_process_frames(frames_data)

        # Save metadata
        self._save_faces_metadata(faces_csv_data)

        # Log statistics
        self._log_face_extraction_stats(stats, len(frames_data))

        return faces_csv_data

    def _load_frames_metadata(self) -> List[Dict[str, str]]:
        """Load frames metadata from CSV file."""
        metadata_path = os.path.join(self.frame_dir, self.FRAMES_METADATA_FILE)

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Frames metadata not found at {metadata_path}. "
                "Please run extract_frame() first."
            )

        with open(metadata_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)

    def _parallel_process_frames(
        self,
        frames_data: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Process frames in parallel to extract faces."""
        process_func = partial(
            self._process_single_frame,
            faces_dir=str(self.faces_dir),
            config=self.config
        )

        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_func, frames_data),
                total=len(frames_data),
                desc="Extracting faces",
                unit="frame"
            ))

        # Collect statistics
        faces_csv_data = []
        total_faces = 0
        frames_with_no_faces = 0

        for result in results:
            if result:
                faces_csv_data.extend(result)
                total_faces += len(result)
            else:
                frames_with_no_faces += 1

        stats = {
            'total_faces': total_faces,
            'frames_with_no_faces': frames_with_no_faces
        }

        return faces_csv_data, stats

    @staticmethod
    def _process_single_frame(
        frame_data: Dict[str, str],
        faces_dir: str,
        config: ProcessingConfig
    ) -> List[Dict[str, Any]]:
        """
        Worker function to extract faces from a single frame.

        Args:
            frame_data: Dictionary containing frame metadata
            faces_dir: Directory to save extracted faces
            config: Processing configuration

        Returns:
            List of face metadata dictionaries
        """
        frame_path = frame_data['Frame_Path']
        frame_number = int(frame_data['Frame_Number'])
        video_path = frame_data['Video_Path']
        label = frame_data['Label']
        dataset = frame_data['Dataset']

        if not os.path.exists(frame_path):
            return []

        try:
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                return []

            # Initialize detectors
            face_detector = FaceDetector(
                min_face_size=config.min_face_size,
                confidence_threshold=config.confidence_threshold,
                only_keep_top=config.only_keep_top_face
            )
            face_extractor = FaceExtractor(
                target_size=config.face_target_size,
                normalization=config.face_normalization,
                padding=config.face_padding
            )

            # Detect faces
            detections = face_detector.detect_faces(frame, return_landmarks=False)

            if not detections:
                return []

            # Extract and save each face
            faces_data = []
            for idx, detection in enumerate(detections):
                face_metadata = DatasetBuilder._extract_and_save_face(
                    frame=frame,
                    detection=detection,
                    frame_number=frame_number,
                    frame_path=frame_path,
                    video_path=video_path,
                    label=label,
                    dataset=dataset,
                    idx=idx,
                    faces_dir=faces_dir,
                    face_extractor=face_extractor
                )
                faces_data.append(face_metadata)

            return faces_data

        except Exception as e:
            logger = logging.getLogger("DatasetBuilder.Worker")
            logger.error(f"Error processing frame {frame_path}: {str(e)}")
            return []

    @staticmethod
    def _extract_and_save_face(
        frame: Any,
        detection: Dict[str, Any],
        frame_number: int,
        frame_path: str,
        video_path: str,
        label: str,
        dataset: str,
        idx: int,
        faces_dir: str,
        face_extractor: FaceExtractor
    ) -> Dict[str, Any]:
        """Extract, save, and create metadata for a single face."""
        bbox = detection['bbox']
        confidence = detection['confidence']

        # Extract and preprocess face
        preprocessed_face, _ = face_extractor.extract_and_preprocess(
            frame, bbox, normalize=True
        )

        # Determine output path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        face_output_dir = os.path.join(faces_dir, video_name)
        os.makedirs(face_output_dir, exist_ok=True)

        face_filename = f"face_frame_{frame_number:06d}_{idx:02d}.jpg"
        face_path = os.path.join(face_output_dir, face_filename)

        # Save face
        face_extractor.save(preprocessed_face, face_path, denormalize=True)

        # Create metadata
        return {
            "Face_Path": face_path,
            "Frame_Path": frame_path,
            "Video_Path": video_path,
            "Label": label,
            "Frame_Number": frame_number,
            "Dataset": dataset,
            "Confidence": confidence,
            "BBox_X": bbox[0],
            "BBox_Y": bbox[1],
            "BBox_Width": bbox[2],
            "BBox_Height": bbox[3],
            "Face_Width": face_extractor.target_size[0],
            "Face_Height": face_extractor.target_size[1]
        }

    def _save_faces_metadata(self, csv_data: List[Dict[str, Any]]) -> None:
        """Save face extraction metadata to CSV file."""
        metadata_path = os.path.join(self.faces_dir, self.FACES_METADATA_FILE)

        self.logger.info(f"Saving faces metadata to {metadata_path}...")

        with open(metadata_path, 'w', newline='') as csvfile:
            if csv_data:
                writer = csv.DictWriter(csvfile, fieldnames=self.FACE_CSV_FIELDS)
                writer.writeheader()
                writer.writerows(csv_data)

        self.logger.info(f"✓ Metadata saved to {metadata_path}")

    def _log_face_extraction_stats(
        self,
        stats: Dict[str, int],
        total_frames: int
    ) -> None:
        """Log face extraction statistics."""
        self.logger.info("✓ Face extraction complete!")
        self.logger.info(
            f"✓ {stats['total_faces']} faces extracted from {total_frames} frames"
        )
        self.logger.info(
            f"✓ {stats['frames_with_no_faces']} frames had no detectable faces"
        )
