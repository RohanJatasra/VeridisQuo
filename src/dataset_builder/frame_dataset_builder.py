"""
FrameDatasetBuilder module for extracting frames from videos.
"""

from __future__ import annotations

import os
import logging
from typing import List, Tuple
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm

from preprocessing.frames_extractor import FramesExtractor
from data.schemas import CSVItem, FrameCSVItem
from utils.csv_services import CSVService


class FrameDatasetBuilder:
    """
    Handles frame extraction from videos in parallel.

    This class is responsible for:
    - Extracting frames from video files
    - Saving frame images to disk
    - Generating and saving frame metadata CSV
    """

    FRAMES_METADATA_FILE = "frames_metadata.csv"

    def __init__(
        self,
        data_folder: str,
        frame_dir: str,
        videos_path: dict,
        num_workers: int,
        fps: int = 1
    ):
        """
        Initialize FrameDatasetBuilder.

        Args:
            data_folder: Base folder containing video files
            frame_dir: Directory to save extracted frames
            videos_path: Dictionary mapping dataset name to list of CSVItem
            num_workers: Number of parallel workers
            fps: Frames per second to extract
        """
        self.data_folder = data_folder
        self.frame_dir = frame_dir
        self.videos_path = videos_path
        self.num_workers = num_workers
        self.fps = fps
        self.logger = logging.getLogger("FrameDatasetBuilder")

    def extract_frames(self) -> List[FrameCSVItem]:
        """
        Extract frames from all videos in parallel.

        Returns:
            List of FrameCSVItem instances
        """
        self.logger.info("Starting frame extraction...")

        # Collect all videos
        all_videos = self._collect_videos()
        self.logger.info(f"Found {len(all_videos)} videos to process")

        # Process in parallel
        csv_data = self._parallel_process_videos(all_videos)

        # Save metadata
        self._save_metadata(csv_data)

        self.logger.info(
            f"✓ Frame extraction complete! "
            f"{len(csv_data)} frames extracted from {len(all_videos)} videos"
        )

        return csv_data

    def _collect_videos(self) -> List[Tuple[str, CSVItem]]:
        """Collect all videos from loaded CSV files."""
        videos = []
        for csv_name, video_list in self.videos_path.items():
            for video in video_list:
                videos.append((csv_name, video))
        return videos

    def _parallel_process_videos(
        self,
        videos: List[Tuple[str, CSVItem]]
    ) -> List[FrameCSVItem]:
        """Process videos in parallel to extract frames."""
        process_func = partial(
            self._process_single_video,
            data_folder=str(self.data_folder),
            frame_dir=str(self.frame_dir),
            fps=self.fps
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
        video_data: Tuple[str, CSVItem],
        data_folder: str,
        frame_dir: str,
        fps: int
    ) -> List[FrameCSVItem]:
        """
        Worker function to extract frames from a single video.

        Args:
            video_data: Tuple of (csv_name, video CSVItem object)
            data_folder: Base data folder path
            frame_dir: Directory to save extracted frames
            fps: Frames per second to extract

        Returns:
            List of FrameCSVItem instances
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

                # Create relative path (relative to frame_dir)
                relative_frame_path = os.path.join(video_name, frame_filename)

                frame_item = FrameCSVItem(
                    Frame_Path=relative_frame_path,
                    Video_Path=video_path,
                    Label=video.Label,
                    Width=extractor.width,
                    Height=extractor.height,
                    Frame_Number=metadata['frame_number'],
                    Timestamp=metadata['timestamp'],
                    Dataset=csv_name.replace('.csv', '')
                )
                frames_data.append(frame_item)

            return frames_data

        except Exception as e:
            logger = logging.getLogger("FrameDatasetBuilder.Worker")
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return []

    def _save_metadata(self, csv_data: List[FrameCSVItem]) -> None:
        """Save frame extraction metadata to CSV file."""
        metadata_path = os.path.join(self.frame_dir, self.FRAMES_METADATA_FILE)

        self.logger.info(f"Saving metadata to {metadata_path}...")

        CSVService.save_csv(metadata_path, csv_data, FrameCSVItem)

        self.logger.info("✓ Metadata saved")
