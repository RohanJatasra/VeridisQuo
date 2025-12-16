import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any, Literal
import logging
import warnings

# Try importing torchvision for fast video decoding
try:
    import torch
    from torchvision.io import VideoReader
    # Test if VideoReader actually works (requires pyav)
    try:
        # Try to instantiate VideoReader to verify pyav is available
        TORCHVISION_AVAILABLE = True
    except Exception:
        TORCHVISION_AVAILABLE = False
except ImportError:
    TORCHVISION_AVAILABLE = False
    torch = None

class FramesExtractor:
    """Extract frames from video files with configurable sampling rate, extracted frame data
    and metadata (timestamp, frame number).

    Supports multiple backends for optimal performance:
    - 'torchvision': Fastest, uses GPU acceleration when available (default if available)
    - 'opencv': Reliable fallback, CPU-based with optimized sequential reading
    - 'auto': Automatically selects best available backend
    """

    def __init__(self,
        video_path: str,
        nb_fps: Optional[int] = None,
        backend: Literal['auto', 'torchvision', 'opencv'] = 'auto',
        use_gpu: bool = True,
    ):
        """Initialize VideoExtractor
        Parameters:
            video_path (str): path to the video file
            nb_fps (int, optional): number of frames to extract per second.
            backend (str): Backend to use ('auto', 'torchvision', 'opencv'). Default 'auto'.
            use_gpu (bool): Whether to use GPU acceleration when available. Default True.
        Raises:
            FileNotFoundError: if video file doesn't exist
            ValueError: if video fail to open
        """
        self.video_path: Path = Path(video_path)
        """Video file path"""
        self.nb_fps: Optional[int] = nb_fps
        """Number of frames to extract per second"""
        self.logger: logging.Logger = logging.getLogger("preprocessing.FramesExtractor")
        """Logger instance for the FramesExtractor class"""
        if not self.video_path.exists():
            self.logger.fatal(f"Video file not found: {self.video_path}")
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Select backend
        self.backend = self._select_backend(backend)

        # Set GPU flag only if using torchvision and CUDA is available
        if self.backend == 'torchvision' and TORCHVISION_AVAILABLE:
            self.use_gpu = use_gpu and torch.cuda.is_available()
        else:
            self.use_gpu = False

        self.logger.debug(f"Using backend: {self.backend}, GPU: {self.use_gpu}")

        # Initialize video properties using OpenCV (most reliable for metadata)
        self._video_props: cv2.VideoCapture = cv2.VideoCapture(str(self.video_path))
        """Captures video properties"""

        if not self._video_props.isOpened():
            self.logger.fatal(f"Could not open video file: {self.video_path}")
            raise ValueError(f"Could not open video file: {self.video_path}")

        self.video_fps = self._video_props.get(cv2.CAP_PROP_FPS)
        """Fps of the video"""
        try:
            assert self.video_fps > 0, "Video has 0 FPS"

        except AssertionError as e:
            self._video_props.release()
            self.logger.fatal(f"Invalid parameters: {self.video_fps} for video {self.video_path}")
            raise ValueError(f"Invalid parameters: {self.video_fps} for video {self.video_path}")

        self.total_frames = int(self._video_props.get(cv2.CAP_PROP_FRAME_COUNT))
        """Total number of frames in the video"""
        try:
            assert self.total_frames > 0, "Video has 0 frames"

        except AssertionError as e:
            self._video_props.release()
            self.logger.fatal(f"Invalid parameters: {self.total_frames} for video {self.video_path}")
            raise ValueError(f"Invalid parameters: {self.total_frames} for video {self.video_path}")

        self.width = int(self._video_props.get(cv2.CAP_PROP_FRAME_WIDTH))
        """Width of the video frames"""
        self.height = int(self._video_props.get(cv2.CAP_PROP_FRAME_HEIGHT))
        """Height of the video frames"""
        self.duration = self.total_frames / self.video_fps if self.video_fps > 0 else 0
        """Duration of the video in seconds"""

        if self.nb_fps is not None and self.nb_fps > 0:
            self.frame_interval = max(1, int(self.video_fps / self.nb_fps))
        else:
            self.frame_interval = 1  # every frame extracted by default

        # For torchvision backend, release OpenCV capture as we'll use torchvision
        if self.backend == 'torchvision':
            self._video_props.release()
            self._video_props = None

    def _select_backend(self, backend: str) -> str:
        """Select the best available backend."""
        if backend == 'torchvision':
            if not TORCHVISION_AVAILABLE:
                self.logger.warning("torchvision video backend not available (requires pyav), falling back to opencv")
                return 'opencv'
            return 'torchvision'
        elif backend == 'opencv':
            return 'opencv'
        else:  # auto
            # Default to OpenCV for reliability
            # Users can explicitly request 'torchvision' if they have pyav installed
            if TORCHVISION_AVAILABLE:
                self.logger.debug("torchvision available but using opencv by default for stability")
            return 'opencv'
        
    def extract_frames(self) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """Extract frames from the video using the selected backend.
        Yields:
            Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]: frames and associated metadata
            - np.ndarray of shape (height, width, 3), frames in BGR format
            - Dict[str, Any] with keys 'frame_id', 'timestamp', 'frame_number'
        Raises:
            RuntimeError: if frame reading fails
        """
        if self.backend == 'torchvision':
            yield from self._extract_frames_torchvision()
        else:
            yield from self._extract_frames_opencv_optimized()

    def _extract_frames_torchvision(self) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """Extract frames using torchvision backend (fastest method)."""
        try:
            if not TORCHVISION_AVAILABLE:
                raise ImportError("torchvision not available")

            # Read entire video or use VideoReader for streaming
            # For memory efficiency with large videos, use VideoReader
            reader = VideoReader(str(self.video_path), "video")
            reader.set_current_stream("video:0")

            frame_count = 0
            extracted_count = 0

            for frame_data in reader:
                if frame_count % self.frame_interval == 0:
                    # Convert from torch tensor (C, H, W) to numpy (H, W, C)
                    frame_tensor = frame_data['data']

                    # Move to CPU if on GPU
                    if frame_tensor.is_cuda:
                        frame_tensor = frame_tensor.cpu()

                    # Convert from (C, H, W) to (H, W, C) and to numpy
                    frame = frame_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

                    # Convert RGB to BGR (OpenCV format)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    timestamp = frame_count / self.video_fps if self.video_fps > 0 else 0

                    metadata = {
                        "frame_id": frame_count,
                        "frame_number": frame_count,
                        "timestamp": timestamp,
                        "video_fps": self.video_fps,
                        "resolution": (self.width, self.height)
                    }

                    yield frame, metadata
                    extracted_count += 1

                frame_count += 1

                # Safety check
                if frame_count >= self.total_frames:
                    break

            self.logger.info(
                f"- extract_frames - torchvision extraction complete: "
                f"{extracted_count} frames extracted from {frame_count} total frames."
            )

        except Exception as e:
            self.logger.warning(f"torchvision backend failed: {str(e)[:100]}, falling back to OpenCV")
            # Fallback to OpenCV
            self.backend = 'opencv'
            if self._video_props is None:
                self._video_props = cv2.VideoCapture(str(self.video_path))
            yield from self._extract_frames_opencv_optimized()

    def _extract_frames_opencv_optimized(self) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """Extract frames using optimized OpenCV sequential reading (no repositioning)."""
        frame_count = 0
        extracted_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 10

        try:
            # Reset video to beginning
            self._video_props.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while frame_count < self.total_frames:
                # Read frame sequentially (MUCH faster than repositioning)
                is_read, frame = self._video_props.read()

                if not is_read:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.warning(
                            f"- extract_frames - too many consecutive failures ({consecutive_failures}), "
                            f"stopping at frame {frame_count}."
                        )
                        break
                    frame_count += 1
                    continue

                # Only yield frames according to frame_interval
                if frame_count % self.frame_interval == 0:
                    consecutive_failures = 0

                    timestamp = frame_count / self.video_fps if self.video_fps > 0 else 0

                    metadata = {
                        "frame_id": frame_count,
                        "frame_number": frame_count,
                        "timestamp": timestamp,
                        "video_fps": self.video_fps,
                        "resolution": (self.width, self.height)
                    }

                    yield frame, metadata
                    extracted_count += 1

                frame_count += 1

            self.logger.info(
                f"- extract_frames - opencv extraction complete: "
                f"{extracted_count} frames extracted from {frame_count} total frames."
            )

        except Exception as e:
            self.logger.error(f"- extract_frames_opencv - {e}")
            raise RuntimeError(f"Error during frame extraction: {e}")
        
    def get_frame_at(self, frame_index: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Extract a specific frame by frame number.

        Note: This method uses random access which is slower than sequential reading.
        For extracting multiple frames, use extract_frames() instead.

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
            self.logger.error(f"- get_frame_at - {e}")
            raise IndexError(f"Error getting frame at index {frame_index}: {e}")

        if self.backend == 'torchvision':
            return self._get_frame_at_torchvision(frame_index)
        else:
            return self._get_frame_at_opencv(frame_index)

    def _get_frame_at_opencv(self, frame_index: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get frame using OpenCV (requires repositioning, slower for random access)."""
        if self._video_props is None:
            self._video_props = cv2.VideoCapture(str(self.video_path))

        self._video_props.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        is_read, frame = self._video_props.read()

        if not is_read:
            self.logger.debug(f"- get_frame_at - failed to read frame {frame_index}")
            return None

        timestamp = frame_index / self.video_fps if self.video_fps > 0 else 0

        metadata = {
            "frame_id": frame_index,
            "frame_number": frame_index,
            "timestamp": timestamp,
            "video_fps": self.video_fps,
            "resolution": (self.width, self.height)
        }

        return frame, metadata

    def _get_frame_at_torchvision(self, frame_index: int) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get frame using torchvision (seeks to specific frame)."""
        try:
            reader = VideoReader(str(self.video_path), "video")
            reader.set_current_stream("video:0")

            # Seek to frame
            reader.seek(frame_index / self.video_fps)

            # Get next frame
            frame_data = next(reader)
            frame_tensor = frame_data['data']

            # Move to CPU if on GPU
            if frame_tensor.is_cuda:
                frame_tensor = frame_tensor.cpu()

            # Convert from (C, H, W) to (H, W, C) and to numpy
            frame = frame_tensor.permute(1, 2, 0).numpy().astype(np.uint8)

            # Convert RGB to BGR (OpenCV format)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            timestamp = frame_index / self.video_fps if self.video_fps > 0 else 0

            metadata = {
                "frame_id": frame_index,
                "frame_number": frame_index,
                "timestamp": timestamp,
                "video_fps": self.video_fps,
                "resolution": (self.width, self.height)
            }

            return frame, metadata

        except Exception as e:
            self.logger.debug(f"- get_frame_at_torchvision - failed to read frame {frame_index}: {e}")
            return None

    def get_video_info(self) -> Dict[str, Any]:
        """Get video information.
        Returns:
            Dict[str, Any]: dictionary with video properties
        """
        return {
            "path": str(self.video_path),
            "fps": self.video_fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration": self.duration,
            "frame_interval": self.frame_interval,
            "backend": self.backend,
            "gpu_enabled": self.use_gpu
        }

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
            assert Path.exists(Path(output_path).parent), "Output directory does not exist"
        
        except AssertionError as e:
            self.logger.error(f"- save_frame - {e}")
            raise ValueError(f"Invalid parameters for save_frame: {e}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path), frame)

        if not success:
            raise RuntimeError(f"Failed to save frame to {output_path}")

    def __del__(self):
        """Destructor - ensure resources are released."""
        if hasattr(self, '_video_props') and self._video_props is not None:
            try:
                self._video_props.release()
            except:
                pass
