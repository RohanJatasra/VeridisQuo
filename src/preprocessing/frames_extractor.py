import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any
import logging
from .base_frames_extractor import BaseFramesExtractor


class FramesExtractor(BaseFramesExtractor):
    """Extract frames from video files using OpenCV with configurable sampling rate.

    Provides frame data and metadata (timestamp, frame number).
    Uses OpenCV's VideoCapture for frame extraction.
    """

    def __init__(self, video_path: str, nb_fps: int):
        """Initialize FramesExtractor.
        Parameters:
            video_path (str): path to the video file
            nb_fps (int): number of frames to extract per second
        Raises:
            FileNotFoundError: if video file doesn't exist
            ValueError: if video fail to open
        """
        super().__init__(
            video_path=video_path,
            nb_fps=nb_fps,
            logger_name="/".join(__file__.split("/")[-2:])
        )

        #open video with OpenCV
        try:
            self._video_props: cv2.VideoCapture = cv2.VideoCapture(str(self.video_path))
            """OpenCV VideoCapture object for reading frames"""

            assert self._video_props.isOpened(), f"Could not open video file: {self.video_path}"

            #set video properties from OpenCV
            self.video_fps = self._video_props.get(cv2.CAP_PROP_FPS)
            assert self.video_fps > 0, "Video has 0 FPS"

            self.total_frames = int(self._video_props.get(cv2.CAP_PROP_FRAME_COUNT))
            assert self.total_frames > 0, "Video has 0 frames"

            self.width = int(self._video_props.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._video_props.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0

        except AssertionError as e:
            self._video_props.release()
            err_msg = f"Invalid parameters: {e}"
            self.logger.fatal(err_msg)
            raise ValueError(err_msg)

        self._calculate_frame_interval()
        
    def extract_frames(self) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """Extract frames from the video.
        Yields:
            Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]: frames and associated metadata
            - np.ndarray of shape (height, width, 3), frames in BGR format
            - Dict[str, Any] with keys 'frame_id', 'timestamp', 'frame_number'
        Raises:
            RuntimeError: if frame reading fails
            IndexError: if frame_index is out of range
        """
        frame_count = 0
        extracted_count = 0

        try:
            #reset video to beginning
            self._video_props.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while frame_count < self.total_frames:
                #we only extract every n frame according to frame_interval
                if frame_count % self.frame_interval == 0:
                    extracted = self.get_frame_at(frame_count)

                    if extracted is None:
                        self.logger.info(f"Reached end of video at frame {frame_count}.")
                        break

                    yield extracted

                    extracted_count += 1

                frame_count += 1

            self.logger.info(f"Frame extraction complete: {extracted_count} frames extracted from {frame_count} total frames.")

        except Exception as e:
            err_msg = f"Error during frame extraction: {e}"
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)
        
    def get_frame_at(self, frame_index: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Extract a specific frame by frame number.
        Parameters:
            frame_index (int): frame index to extract
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: (frame, metadata) or None if frame doesn't exist
        Raises:
            IndexError: if frame_index is out of range
        """
        try:
            assert frame_index >= 0 and frame_index < self.total_frames, f"Frame index {frame_index} out of range [0, {self.total_frames})"

        except AssertionError as e:
            err_msg = f"Error getting frame at index {frame_index}: {e}"
            self.logger.error(err_msg)
            raise IndexError(err_msg)

        self._video_props.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        is_read, frame = self._video_props.read()

        if not is_read:
            self.logger.debug(f"Failed to read frame {frame_index}")
            return None

        #use base class method to build metadata
        metadata = self._build_metadata(frame_index)

        return frame, metadata

    def save_frame(self, frame: np.ndarray, output_path: str) -> None:
        """Save a frame to the provided location.
        Parameters:
            frame (np.ndarray): frame to save (numpy array)
            output_path (str): path where to save the frame
        Raises:
            ValueError: if parameters are invalid
            RuntimeError: if saving fails
        """
        try:
            assert frame is not None and frame.size > 0, "Invalid frame to save"
            output_path = Path(output_path)
            assert output_path.parent.exists(), "Output directory does not exist"
        
        except (AssertionError, TypeError) as e:
            self.logger.error(e)
            raise ValueError(f"Invalid parameters for save_frame: {e}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            assert cv2.imwrite(str(output_path), frame), "OpenCV failed to save frame"

        except Exception as e:
            err_mmsg = f"Failed to save frame to {output_path}: {e}"
            self.logger.error(err_mmsg)
            raise RuntimeError(err_mmsg)

    def __del__(self):
        """Destructor - ensure resources are released."""
        if self._video_props is not None:
            self._video_props.release()
