from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import logging
from multiprocessing import cpu_count
from typing import Tuple

from core.csv_schemas import CSVItem, FrameCSVItem, FaceCSVItem, DatasetSplit, PreprocessCSVItem
from dataset_builder.face_forensic import FaceForensicDataset
from dataset_builder.frame_dataset_builder import FrameDatasetBuilder
from dataset_builder.face_dataset_builder import FaceDatasetBuilder
from dataset_builder.dataset_balancer import DatasetBalancer
from dataset_builder.dataset_splitter import DatasetSplitter
from utils.csv_services import CSVService
import shutil
from pathlib import Path


@dataclass
class ProcessingConfig:
    """Configuration for face detection and extraction."""

    # Frame extraction
    frames_per_second: int

    # Face detection
    min_face_size: int
    confidence_threshold: float
    only_keep_top_face: bool

    # Face extraction
    face_target_size: Tuple[int, int]
    face_normalization: str 
    face_padding: int # pixels to add around face bbox


class DatasetBuilder:
    """
    Main orchestrator for dataset building.

    Delegates frame extraction to FrameDatasetBuilder and
    face extraction to FaceDatasetBuilder.

    This class provides a simple, unified interface for the complete
    dataset preprocessing pipeline.
    """
    
    def __init__(self,
        num_workers: Optional[int],
        device: str,
        frame_per_second: int,
        min_face_size: int,
        confidence_threshold: float,
        only_keep_top_face: bool,
        face_target_size: Tuple[int, int],
        face_normalization: str,
        face_padding: int,
        csv_folder: str,
        data_folder: str,
        frame_dir: str,
        faces_dir: str
    ):
        """
        Initialize DatasetBuilder.

        Args:
            mode: Dataset mode - 'train' or 'test'
            num_workers: Number of parallel workers. Defaults to CPU count.
            config: Processing configuration. Defaults to ProcessingConfig().
            device: Device to use for face detection (cpu or cuda)

        Raises:
            ValueError: If mode is not 'train' or 'test'
        """
        self.logger: logging.Logger = logging.getLogger("/".join(__file__.split("/")[-2:]))

        self.num_workers: int = num_workers if num_workers is not None else cpu_count()
        self.device: str = device

        self.config: ProcessingConfig = ProcessingConfig(
            frames_per_second=frame_per_second,
            min_face_size=min_face_size,
            confidence_threshold=confidence_threshold,
            only_keep_top_face=only_keep_top_face,
            face_target_size=face_target_size,
            face_normalization=face_normalization,
            face_padding=face_padding
        )
        
        self.csv_folder = csv_folder
        self.data_folder = data_folder
        self.frame_dir = frame_dir
        self.faces_dir = faces_dir

        # Initialize dataset and logger
        self.dataset = FaceForensicDataset(self.csv_folder)
        self.videos_path: Dict[str, List[CSVItem]] = self.dataset.load_all_csvs()

        self.logger.info(f"Initialized DatasetBuilder with {self.num_workers} workers")

        # Initialize sub-builders (lazy initialization)
        self._frame_builder: Optional[FrameDatasetBuilder] = None
        self._face_builder: Optional[FaceDatasetBuilder] = None
        
    @property
    def frame_builder(self) -> FrameDatasetBuilder:
        """Get or create FrameDatasetBuilder instance."""
        if self._frame_builder is None:
            self._frame_builder = FrameDatasetBuilder(
                data_folder=self.data_folder,
                frame_dir=self.frame_dir,
                videos_path=self.videos_path,
                num_workers=self.num_workers,
                fps=self.config.frames_per_second,
                device=self.device,
                batch_size=32,
                num_save_threads=4
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
                face_padding=self.config.face_padding,
                device=self.device
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

    def build_complete_dataset(self,
        name: str,
        balance: bool,
        balance_method: str,
        split: bool,
        train_ratio: float,
        test_ratio: float,
        eval_ratio: float,
        save_preprocessed: bool,
        random_state: Optional[int]
    ) -> Union[Tuple[List[FrameCSVItem], List[FaceCSVItem]], Tuple[List[FrameCSVItem], DatasetSplit]]:
        """
        Run the complete pipeline: extract frames, faces, balance, split, and save.

        Pipeline steps:
        1. Extract frames from videos → saved to FRAME_DIR
        2. Extract faces from frames → saved to FACES_DIR
        3. Balance FAKE/REAL classes (optional)
        4. Split into train/test/eval sets (optional)
        5. Save preprocessed dataset (optional) → saved to PREPROCESSED_DATASET_DIR

        Args:
            balance: Whether to balance FAKE/REAL classes (default: True)
            balance_method: Method for balancing - 'undersample' or 'oversample' (default: 'undersample')
            split: Whether to split into train/test/eval sets (default: True)
            train_ratio: Proportion for training set (default: 0.7)
            test_ratio: Proportion for test set (default: 0.15)
            eval_ratio: Proportion for eval set (default: 0.15)
            save_preprocessed: Whether to save preprocessed dataset to disk (default: True)
            random_state: Random seed for reproducibility (default: 42)

        Returns:
            If split=True: Tuple of (frames_data, dataset_split)
            If split=False: Tuple of (frames_data, faces_data)
        """
        self.logger.info("Starting complete dataset build pipeline...")

        # Step 1: Extract frames
        self.logger.info("Step 1/5: Extracting frames from videos...")
        frames_data = self.extract_frame()
        self.logger.info(f"Extracted {len(frames_data)} frames")

        # Step 2: Extract faces
        self.logger.info("Step 2/5: Extracting faces from frames...")
        faces_data = self.extract_face()
        self.logger.info(f"Extracted {len(faces_data)} faces")

        # Step 3: Balance dataset (optional)
        if balance:
            self.logger.info(f"Step 3/5: Balancing dataset using {balance_method}...")
            balancer = DatasetBalancer(random_state=random_state)

            # Log initial distribution
            distribution = balancer.get_class_distribution(faces_data)
            self.logger.info(
                f"Initial distribution - FAKE: {distribution['fake']['count']}, "
                f"REAL: {distribution['real']['count']} "
                f"(ratio: {distribution['ratio']:.2f})"
            )

            faces_data = balancer.balance(faces_data, method=balance_method)
            self.logger.info("Dataset balanced")
        else:
            self.logger.info("Step 3/5: Skipping balancing (balance=False)")

        # Step 4: Split dataset (optional)
        if split:
            self.logger.info("Step 4/5: Splitting dataset into train/test/eval...")
            splitter = DatasetSplitter(random_state=random_state)

            dataset_split = splitter.split(
                faces_data,
                train_ratio=train_ratio,
                test_ratio=test_ratio,
                eval_ratio=eval_ratio
            )

            # Validate split
            if splitter.validate_split(dataset_split, faces_data):
                self.logger.info("Split validation passed")
            else:
                self.logger.warning("Split validation failed!")

            # Step 5: Save preprocessed dataset (optional)
            if save_preprocessed:
                self.logger.info("Step 5/5: Saving preprocessed dataset...")
                self.save_preprocessed_dataset(dataset_split, output_name=name)
                self.logger.info("Preprocessed dataset saved")
            else:
                self.logger.info("Step 5/5: Skipping save (save_preprocessed=False)")

            self.logger.info("Complete dataset build finished!")
            return frames_data, dataset_split
        else:
            self.logger.info("Step 4/5: Skipping split (split=False)")
            self.logger.info("Step 5/5: Skipping save (split=False)")
            self.logger.info("Complete dataset build finished!")
            return frames_data, faces_data

    def save_preprocessed_dataset(self,
        dataset_split: DatasetSplit,
        output_dir: str,
    ) -> None:
        """
        Save the preprocessed dataset split to disk.

        Creates the following structure:
        output_dir/
        ├── train.csv
        ├── test.csv
        ├── eval.csv
        ├── train/
        │   └── face_001.jpg
        ├── test/
        │   └── face_002.jpg
        └── eval/
            └── face_003.jpg

        Args:
            dataset_split: DatasetSplit object containing train/test/eval splits
            output_dir: Output directory (default: PREPROCESSED_DATASET_DIR from config)
        """    
        self.logger.info(f"Saving preprocessed dataset to {output_dir}...")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create split directories (no images subdirectory)
        train_dir = output_dir / "train"
        test_dir = output_dir / "test"
        eval_dir = output_dir / "eval"

        for directory in [train_dir, test_dir, eval_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Save each split
        splits = {
            'train': (dataset_split.train, train_dir),
            'test': (dataset_split.test, test_dir),
            'eval': (dataset_split.eval, eval_dir)
        }

        for split_name, (items, split_dir) in splits.items():
            self.logger.info(f"Saving {split_name} split ({len(items)} items)...")

            # Copy face images and update paths to be relative
            updated_items = []
            # Track filename usage to handle duplicates from oversampling
            filename_counter: Dict[str, int] = {}

            for item in items:
                # Reconstruct absolute path from relative path
                # item.Face_Path is relative to faces_dir
                source_path = Path(self.faces_dir) / item.Face_Path
                if source_path.exists():
                    # Generate unique filename to handle duplicates
                    base_name = source_path.stem
                    extension = source_path.suffix

                    if base_name not in filename_counter:
                        filename_counter[base_name] = 0
                        unique_filename = f"{base_name}{extension}"
                    else:
                        filename_counter[base_name] += 1
                        unique_filename = f"{base_name}_{filename_counter[base_name]}{extension}"

                    # Copy with unique filename
                    dest_path = split_dir / unique_filename
                    shutil.copy2(source_path, dest_path)

                    # Create relative path (relative to output_dir)
                    relative_path = f"{split_name}/{unique_filename}"

                    # Create updated item with PreprocessCSVItem (simplified schema)
                    updated_item = PreprocessCSVItem(
                        Face_Path=relative_path,
                        Label=item.Label,
                        Frame_Number=item.Frame_Number,
                        Dataset=item.Dataset,
                        Confidence=item.Confidence,
                        BBox_X=item.BBox_X,
                        BBox_Y=item.BBox_Y,
                        BBox_Width=item.BBox_Width,
                        BBox_Height=item.BBox_Height,
                        Face_Width=item.Face_Width,
                        Face_Height=item.Face_Height
                    )
                    updated_items.append(updated_item)
                else:
                    self.logger.warning(f"Face image not found: {source_path}")

            # Save metadata CSV directly in output_dir with split name
            metadata_path = output_dir / f"{split_name}.csv"
            CSVService.save_csv(str(metadata_path), updated_items, PreprocessCSVItem)

            self.logger.info(f"Saved {split_name}: {len(updated_items)} images + {split_name}.csv")

        # Validate dataset integrity
        self._validate_dataset_integrity(output_dir)

        self.logger.info(f"Preprocessed dataset saved to {output_dir}")

    def _validate_dataset_integrity(self, output_dir: Path) -> None:
        """
        Validate that CSV entries match the actual files in each split directory.

        Args:
            output_dir: Path to the preprocessed dataset directory

        Raises:
            ValueError: If any split has mismatched counts between CSV and files
        """
        self.logger.info("Validating dataset integrity...")

        splits = ['train', 'test', 'eval']
        validation_errors = []

        for split_name in splits:
            csv_path = output_dir / f"{split_name}.csv"
            split_dir = output_dir / split_name

            # Load CSV entries using CSVService
            csv_items: List[PreprocessCSVItem] = []
            if csv_path.exists():
                csv_items = CSVService.load_csv(str(csv_path), PreprocessCSVItem)
            csv_count = len(csv_items)

            # Count image files in directory
            file_count = 0
            if split_dir.exists():
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
                file_count = sum(
                    1 for f in split_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in image_extensions
                )

            # Compare counts
            if csv_count == file_count:
                self.logger.info(f"{split_name}: {csv_count} CSV entries = {file_count} files")
            else:
                error_msg = (f"{split_name}: mismatch - {csv_count} CSV entries vs {file_count} files")
                self.logger.error(error_msg)
                validation_errors.append(error_msg)

        if validation_errors:
            raise ValueError(f"Dataset integrity validation failed:\n" + "\n".join(validation_errors))
