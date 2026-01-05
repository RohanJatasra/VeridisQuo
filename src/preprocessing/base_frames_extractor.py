import logging
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseFramesExtractor(ABC):
    """Abstract base class for frame extractors.

    Defines common interface and shared attributes for all frame extraction implementations.
    Subclasses must implement extract_frames() and get_frame_at() methods.
    """

    def __init__(self, video_path: str, nb_fps: int, logger_name: str):
        """Initialize BaseFramesExtractor with common attributes.
        Parameters:
            video_path (str): path to the video file
            nb_fps (int): number of frames to extract per second
            logger_name (str): name for the logger instance
        Raises:
            FileNotFoundError: if video file doesn't exist
            ValueError: if video path is invalid
        """
        self.video_path: Path = Path(video_path)
        """Video file path"""

        self.nb_fps: int = nb_fps if nb_fps else None #none = all frames
        """Number of frames to extract per second"""

        self.logger: logging.Logger = logging.getLogger(logger_name)
        """Logger instance for the frame extractor"""

        if not self.video_path.exists():
            err_msg = f"Video file not found: {self.video_path}"
            self.logger.fatal(err_msg)
            raise FileNotFoundError(err_msg)

        if not self.video_path.is_file():
            err_msg = f"Provided path is not a file: {self.video_path}"
            self.logger.fatal(err_msg)
            raise ValueError(err_msg)

        self.video_fps: float = 0.0
        """FPS of the video"""

        self.total_frames: int = 0
        """Total number of frames in the video"""

        self.width: int = 0
        """Width of the video frames in pixels"""

        self.height: int = 0
        """Height of the video frames in pixels"""

        self.duration: float = 0.0
        """Duration of the video in seconds"""

        self.frame_interval: int = 1
        """Interval between extracted frames (1 = every frame)"""

    def _calculate_frame_interval(self) -> None:
        """Calculate frame interval based on target FPS.

        Should be called by subclasses after video_fps is set.
        Sets frame_interval to extract approximately nb_fps frames per second.
        """
        if self.nb_fps is not None and self.nb_fps > 0 and self.video_fps > 0:
            self.frame_interval = max(1, int(self.video_fps / self.nb_fps))
        else:
            self.frame_interval = 1  #extract every frame by default

    @abstractmethod
    def extract_frames(self) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """Extract frames from the video.
        Yields:
            Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
            - np.ndarray: frame as BGR numpy array of shape (height, width, 3)
            - Dict[str, Any]: metadata with keys:
                - 'frame_id': unique frame identifier
                - 'frame_number': frame number in video
                - 'timestamp': timestamp in seconds
                - 'video_fps': video frames per second
                - 'resolution': tuple (width, height)
        Raises:
            RuntimeError: if frame extraction fails
        """
        pass

    @abstractmethod
    def get_frame_at(self, frame_index: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Extract a specific frame by frame number.
        Parameters:
            frame_index (int): frame index to extract (0-based)
        Returns:
            Optional[Tuple[np.ndarray, Dict[str, Any]]]: (frame, metadata) or None if frame doesn't exist
        Raises:
            IndexError: if frame_index is out of range
        """
        pass

    def get_video_info(self) -> Dict[str, Any]:
        """Get video information.
        Returns:
            Dict[str, Any]: dictionary with video properties including:
                - path: video file path
                - fps: video frames per second
                - total_frames: total number of frames
                - width: frame width in pixels
                - height: frame height in pixels
                - duration: video duration in seconds
                - frame_interval: interval between extracted frames
        """
        return {
            "path": str(self.video_path),
            "fps": self.video_fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "frame_interval": self.frame_interval
        }

    @abstractmethod
    def save_frame(self, frame: np.ndarray, output_path: str) -> None:
        """Save a frame to disk.
        Parameters:
            frame (np.ndarray): frame to save as numpy array
            output_path (str): path where to save the frame
        Raises:
            ValueError: if parameters are invalid
            RuntimeError: if saving fails
        """
        pass

    def _build_metadata(self, frame_index: int) -> Dict[str, Any]:
        """Build metadata dictionary for a frame.
        Parameters:
            frame_index (int): frame index
        Returns:
            Dict[str, Any]: metadata dictionary
        """
        timestamp = frame_index / self.video_fps if self.video_fps > 0 else 0

        return {
            "frame_id": frame_index,
            "frame_number": frame_index,
            "timestamp": timestamp,
            "video_fps": self.video_fps,
            "resolution": (self.width, self.height)
        }

    @classmethod
    def create_extractor(cls, use_optimized: bool, **kwargs) -> "BaseFramesExtractor":
        """Factory method to create a frames extractor instance.
        If use_optimized is True but the optimized extractor is not available,
        it falls back to the standard extractor.
        Parameters:
            use_optimized (bool): whether to use optimized (PyAV) or standard (OpenCV) implementation
            **kwargs: additional parameters for the extractor. Can include:
            - video_path (str): path to the video file
            - nb_fps (int): number of frames to extract per second
            - device (str): device to use (e.g., 'cpu' or 'cuda')
            - num_threads (int): number of threads for extraction
            - batch_size (int): batch size for extraction
        Returns:
            BaseFramesExtractor: instance of a frames extractor
        """
        if use_optimized:
            from .optimized_frames_extractor import OptimizedFramesExtractor
            if OptimizedFramesExtractor.is_pyav_available():
                return OptimizedFramesExtractor(**kwargs)
        from .frames_extractor import FramesExtractor
        return FramesExtractor(**kwargs)


__all__ = [
    'BaseFramesExtractor',
]
