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
from core.schemas import CSVItem, FrameCSVItem
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
        """Process videos in parallel to extract frames with timeout."""
        process_func = partial(
            self._process_single_video,
            data_folder=str(self.data_folder),
            frame_dir=str(self.frame_dir),
            fps=self.fps
        )

        timeout_seconds = 30  # Timeout per video

        with Pool(processes=self.num_workers) as pool:
            # Submit all tasks asynchronously
            async_results = []
            for video in videos:
                async_result = pool.apply_async(process_func, (video,))
                async_results.append((async_result, video))

            # Collect results with timeout and progress bar
            results = []
            with tqdm(total=len(videos), desc="Extracting frames", unit="video") as pbar:
                for async_result, video_data in async_results:
                    _, video = video_data
                    try:
                        # Wait for result with timeout
                        result = async_result.get(timeout=timeout_seconds)
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(
                            f"Timeout or error processing video {video.File_Path}: {str(e)}"
                        )
                        pbar.update(1)
                        results.append([])  # Empty result for failed video

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
        logger = logging.getLogger("FrameDatasetBuilder.Worker")

        if not os.path.exists(complete_video_path):
            logger.warning(f"Video file not found: {complete_video_path}")
            return []

        extractor = None
        try:
            extractor = FramesExtractor(complete_video_path, nb_fps=fps)
            output_dir = os.path.join(frame_dir, video_name)
            os.makedirs(output_dir, exist_ok=True)

            frames_data = []
            frame_count = 0
            max_frames = 1000  # Safety limit to prevent infinite loops

            for frame, metadata in extractor.extract_frames():
                if frame_count >= max_frames:
                    logger.warning(f"Reached max frame limit ({max_frames}) for {video_path}")
                    break

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
                frame_count += 1

            logger.debug(f"Successfully processed {video_path}: {len(frames_data)} frames")
            return frames_data

        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            return []
        finally:
            # Ensure video capture is released
            if extractor is not None and hasattr(extractor, '_video_props'):
                try:
                    extractor._video_props.release()
                except:
                    pass

    def _save_metadata(self, csv_data: List[FrameCSVItem]) -> None:
        """Save frame extraction metadata to CSV file."""
        metadata_path = os.path.join(self.frame_dir, self.FRAMES_METADATA_FILE)

        self.logger.info(f"Saving metadata to {metadata_path}...")

        CSVService.save_csv(metadata_path, csv_data, FrameCSVItem)

        self.logger.info("✓ Metadata saved")
