"""
DatasetBuilder module - orchestrates frame and face extraction.
"""

from __future__ import annotations

# Suppress TensorFlow verbose output BEFORE any TF imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from multiprocessing import cpu_count
from typing import Tuple

from data.schemas import CSVItem, FrameCSVItem, FaceCSVItem
from data.face_forensic import FaceForensicDataset
from data.frame_dataset_builder import FrameDatasetBuilder
from data.face_dataset_builder import FaceDatasetBuilder

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
    face_padding: int = 20  # pixels to add around face bbox


class DatasetBuilder:
    """
    Main orchestrator for dataset building.

    Delegates frame extraction to FrameDatasetBuilder and
    face extraction to FaceDatasetBuilder.

    This class provides a simple, unified interface for the complete
    dataset preprocessing pipeline.
    """

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
        self.dataset = FaceForensicDataset(self.csv_folder)
        self.videos_path: Dict[str, List[CSVItem]] = self.dataset.load_all_csvs()
        self.logger = logging.getLogger(f"DatasetBuilder[{mode}]")

        self.logger.info(
            f"Initialized DatasetBuilder in {mode} mode with {self.num_workers} workers"
        )

        # Initialize sub-builders (lazy initialization)
        self._frame_builder = None
        self._face_builder = None

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

    @property
    def frame_builder(self) -> FrameDatasetBuilder:
        """Get or create FrameDatasetBuilder instance."""
        if self._frame_builder is None:
            self._frame_builder = FrameDatasetBuilder(
                data_folder=self.data_folder,
                frame_dir=self.frame_dir,
                videos_path=self.videos_path,
                num_workers=self.num_workers,
                fps=self.config.frames_per_second
            )
        return self._frame_builder

    @property
    def face_builder(self) -> FaceDatasetBuilder:
        """Get or create FaceDatasetBuilder instance."""
        if self._face_builder is None:
            self._face_builder = FaceDatasetBuilder(
                frame_dir=self.frame_dir,
                faces_dir=self.faces_dir,
                num_workers=self.num_workers,
                min_face_size=self.config.min_face_size,
                confidence_threshold=self.config.confidence_threshold,
                only_keep_top_face=self.config.only_keep_top_face,
                face_target_size=self.config.face_target_size,
                face_normalization=self.config.face_normalization,
                face_padding=self.config.face_padding
            )
        return self._face_builder

    def extract_frame(self) -> List[FrameCSVItem]:
        """
        Extract frames from all videos in parallel.

        Delegates to FrameDatasetBuilder.

        Returns:
            List of FrameCSVItem instances
        """
        return self.frame_builder.extract_frames()

    def extract_face(self) -> List[FaceCSVItem]:
        """
        Extract faces from frames in parallel.

        Delegates to FaceDatasetBuilder.

        Returns:
            List of FaceCSVItem instances
        """
        return self.face_builder.extract_faces()

    def build_complete_dataset(self) -> Tuple[List[FrameCSVItem], List[FaceCSVItem]]:
        """
        Run the complete pipeline: extract frames then faces.

        Returns:
            Tuple of (frames_data, faces_data)
        """
        self.logger.info("Starting complete dataset build pipeline...")

        # Step 1: Extract frames
        frames_data = self.extract_frame()

        # Step 2: Extract faces
        faces_data = self.extract_face()

        self.logger.info("✓ Complete dataset build finished!")
        return frames_data, faces_data
