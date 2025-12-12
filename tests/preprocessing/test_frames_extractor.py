import pytest
import sys
import os
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "frames_extractor",
    os.path.join(os.path.dirname(__file__), '../..', 'src', 'preprocessing', 'frames_extractor.py')
)
frames_extractor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frames_extractor_module)
FramesExtractor = frames_extractor_module.FramesExtractor

class TestFramesExtractorInit:
    """Tests for FramesExtractor initialization."""

    def test_init_with_zero_fps(self, mocker, tmp_path):
        """Test initialization with video having 0 FPS raises ValueError."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 0,
            cv2.CAP_PROP_FRAME_COUNT: 100
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)

        with pytest.raises(ValueError, match="Invalid parameters"):
            FramesExtractor(str(test_video))

        # Verify resource cleanup (release should be called at least once)
        assert mock_cap.release.call_count >= 1

    def test_init_with_zero_frames(self, mocker, tmp_path):
        """Test initialization with video having 0 frames raises ValueError."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 0
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)

        with pytest.raises(ValueError, match="Invalid parameters"):
            FramesExtractor(str(test_video))

        # Verify resource cleanup (release should be called at least once)
        assert mock_cap.release.call_count >= 1

    def test_init_with_valid_video(self, mocker, tmp_path):
        """Test successful initialization with valid video."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)

        extractor = FramesExtractor(str(test_video))

        assert extractor.video_fps == 30
        assert extractor.total_frames == 100
        assert extractor.width == 1920
        assert extractor.height == 1080
        assert extractor.duration == pytest.approx(100 / 30, 0.01)
        assert extractor.frame_interval == 1  # Default when nb_fps is None

    def test_init_with_nb_fps_parameter(self, mocker, tmp_path):
        """Test initialization with nb_fps parameter sets correct frame_interval."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)

        # Test with nb_fps = 10 (should extract 1 frame every 3 frames)
        extractor = FramesExtractor(str(test_video), nb_fps=10)
        assert extractor.frame_interval == 3  # 30 / 10 = 3

        # Test with nb_fps = 1 (should extract 1 frame every 30 frames)
        extractor = FramesExtractor(str(test_video), nb_fps=1)
        assert extractor.frame_interval == 30  # 30 / 1 = 30

class TestFramesExtractorGetFrameAt:
    """Tests for FramesExtractor.get_frame_at() method."""

    def test_get_frame_at_negative_index(self, mocker, tmp_path):
        """Test get_frame_at with negative index raises IndexError."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = self._setup_mock_video(mocker)

        extractor = FramesExtractor(str(test_video))

        with pytest.raises(IndexError, match="Frame index -1 out of range"):
            extractor.get_frame_at(-1)

    def test_get_frame_at_out_of_range(self, mocker, tmp_path):
        """Test get_frame_at with index >= total_frames raises IndexError."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = self._setup_mock_video(mocker, total_frames=100)

        extractor = FramesExtractor(str(test_video))

        with pytest.raises(IndexError, match="Frame index 100 out of range"):
            extractor.get_frame_at(100)

    def test_get_frame_at_valid_index(self, mocker, tmp_path):
        """Test get_frame_at with valid index returns frame and metadata."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        # Create a mock frame
        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        mock_cap = self._setup_mock_video(mocker)
        mock_cap.read.return_value = (True, mock_frame)

        extractor = FramesExtractor(str(test_video))
        result = extractor.get_frame_at(10)

        assert result is not None
        frame, metadata = result

        # Verify frame
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (1080, 1920, 3)

        # Verify metadata
        assert metadata["frame_id"] == 10
        assert metadata["frame_number"] == 10
        assert metadata["timestamp"] == pytest.approx(10 / 30, 0.01)
        assert metadata["video_fps"] == 30
        assert metadata["resolution"] == (1920, 1080)

        # Verify VideoCapture was positioned correctly
        mock_cap.set.assert_called_with(cv2.CAP_PROP_POS_FRAMES, 10)

    def test_get_frame_at_read_failure(self, mocker, tmp_path):
        """Test get_frame_at returns None when frame read fails."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = self._setup_mock_video(mocker)
        mock_cap.read.return_value = (False, None)

        extractor = FramesExtractor(str(test_video))
        result = extractor.get_frame_at(10)

        assert result is None

    def test_get_frame_at_boundary_frames(self, mocker, tmp_path):
        """Test get_frame_at with first and last frame indices."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap = self._setup_mock_video(mocker, total_frames=100)
        mock_cap.read.return_value = (True, mock_frame)

        extractor = FramesExtractor(str(test_video))

        # Test first frame (index 0)
        result = extractor.get_frame_at(0)
        assert result is not None
        _, metadata = result
        assert metadata["frame_id"] == 0
        assert metadata["timestamp"] == 0

        # Test last frame (index 99)
        result = extractor.get_frame_at(99)
        assert result is not None
        _, metadata = result
        assert metadata["frame_id"] == 99

    def _setup_mock_video(self, mocker, total_frames=100):
        """Helper method to setup mock VideoCapture."""
        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: total_frames,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)
        return mock_cap

class TestFramesExtractorExtractFrames:
    """Tests for FramesExtractor.extract_frames() method."""

    def test_extract_frames_all_frames(self, mocker, tmp_path):
        """Test extract_frames extracts frames when nb_fps is None."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Use total_frames=11 so we can successfully extract frames 0-9
        # Frame 10 will trigger the end condition
        mock_cap = self._setup_mock_video(mocker, total_frames=11)

        # Mock read to return success for first 10 frames, then fail
        read_call_count = [0]
        def mock_read():
            read_call_count[0] += 1
            if read_call_count[0] <= 10:
                return (True, mock_frame.copy())
            return (False, None)

        mock_cap.read.side_effect = mock_read

        extractor = FramesExtractor(str(test_video))

        frames = list(extractor.extract_frames())

        # Should extract 10 frames (indices 0-9)
        assert len(frames) == 10

        for i, (frame, metadata) in enumerate(frames):
            assert isinstance(frame, np.ndarray)
            assert metadata["frame_id"] == i
            assert metadata["frame_number"] == i

    def test_extract_frames_with_interval(self, mocker, tmp_path):
        """Test extract_frames respects frame_interval based on nb_fps."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Use 50 total frames to avoid boundary issues
        # With interval=3, will extract: 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48
        mock_cap = self._setup_mock_video(mocker, total_frames=50)

        # Mock read to return success for first 15 frames, then fail to stop extraction
        read_call_count = [0]
        def mock_read():
            read_call_count[0] += 1
            if read_call_count[0] <= 15:
                return (True, mock_frame.copy())
            return (False, None)

        mock_cap.read.side_effect = mock_read

        # Set nb_fps=10, which should give frame_interval=3 (30fps / 10fps)
        extractor = FramesExtractor(str(test_video), nb_fps=10)

        frames = list(extractor.extract_frames())

        # Should extract 15 frames before read fails
        assert len(frames) == 15

        # Verify first 10 frames are at correct intervals
        expected_frame_ids = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
        for i in range(10):
            assert frames[i][1]["frame_id"] == expected_frame_ids[i]

    def test_extract_frames_generator_behavior(self, mocker, tmp_path):
        """Test extract_frames returns a generator."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap = self._setup_mock_video(mocker, total_frames=5)
        mock_cap.read.return_value = (True, mock_frame)

        extractor = FramesExtractor(str(test_video))

        result = extractor.extract_frames()

        # Verify it's a generator
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_extract_frames_resets_video_position(self, mocker, tmp_path):
        """Test extract_frames resets video to beginning."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap = self._setup_mock_video(mocker, total_frames=6)

        # Mock read to return success for first 5 frames, then fail
        read_call_count = [0]
        def mock_read():
            read_call_count[0] += 1
            if read_call_count[0] <= 5:
                return (True, mock_frame.copy())
            return (False, None)

        mock_cap.read.side_effect = mock_read

        extractor = FramesExtractor(str(test_video))

        # Consume generator
        list(extractor.extract_frames())

        # Verify video was reset to beginning
        mock_cap.set.assert_any_call(cv2.CAP_PROP_POS_FRAMES, 0)

    def test_extract_frames_handles_read_errors(self, mocker, tmp_path):
        """Test extract_frames handles errors during frame reading."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = self._setup_mock_video(mocker)
        # Make get_frame_at raise an exception
        mock_cap.read.side_effect = RuntimeError("Frame read error")

        extractor = FramesExtractor(str(test_video))

        with pytest.raises(RuntimeError, match="Error during frame extraction"):
            list(extractor.extract_frames())

    def _setup_mock_video(self, mocker, total_frames=100):
        """Helper method to setup mock VideoCapture."""
        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: total_frames,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)
        return mock_cap

class TestFramesExtractorGetVideoInfo:
    """Tests for FramesExtractor.get_video_info() method."""

    def test_get_video_info_returns_correct_data(self, mocker, tmp_path):
        """Test get_video_info returns all video properties correctly."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)

        extractor = FramesExtractor(str(test_video), nb_fps=10)
        info = extractor.get_video_info()

        assert info["path"] == str(test_video)
        assert info["fps"] == 30
        assert info["total_frames"] == 100
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["duration"] == pytest.approx(100 / 30, 0.01)
        assert info["frame_interval"] == 3

class TestFramesExtractorSaveFrame:
    """Tests for FramesExtractor.save_frame() method."""

    def test_save_frame_with_valid_parameters(self, mocker, tmp_path):
        """Test save_frame successfully saves a frame."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)
        mock_imwrite = mocker.patch('cv2.imwrite', return_value=True)

        extractor = FramesExtractor(str(test_video))

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        output_dir = tmp_path / "frames"
        output_dir.mkdir()
        output_path = output_dir / "frame_001.jpg"

        # Should not raise any exception
        extractor.save_frame(frame, str(output_path))

        # Verify imwrite was called with correct parameters
        mock_imwrite.assert_called_once_with(str(output_path), frame)

    def test_save_frame_with_none_frame(self, mocker, tmp_path):
        """Test save_frame with None frame raises ValueError."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)

        extractor = FramesExtractor(str(test_video))

        output_path = tmp_path / "frame.jpg"

        with pytest.raises(ValueError, match="Invalid parameters for save_frame"):
            extractor.save_frame(None, str(output_path))

    def test_save_frame_with_empty_frame(self, mocker, tmp_path):
        """Test save_frame with empty frame raises ValueError."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)

        extractor = FramesExtractor(str(test_video))

        empty_frame = np.array([])
        output_path = tmp_path / "frame.jpg"

        with pytest.raises(ValueError, match="Invalid parameters for save_frame"):
            extractor.save_frame(empty_frame, str(output_path))

    def test_save_frame_with_nonexistent_directory(self, mocker, tmp_path):
        """Test save_frame with non-existent output directory raises ValueError."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)

        extractor = FramesExtractor(str(test_video))

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Non-existent nested directory
        output_path = tmp_path / "nonexistent" / "nested" / "frame.jpg"

        with pytest.raises(ValueError, match="Invalid parameters for save_frame"):
            extractor.save_frame(frame, str(output_path))

    def test_save_frame_write_failure(self, mocker, tmp_path):
        """Test save_frame raises RuntimeError when cv2.imwrite fails."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)
        mock_imwrite = mocker.patch('cv2.imwrite', return_value=False)  # Simulate write failure

        extractor = FramesExtractor(str(test_video))

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        output_dir = tmp_path / "frames"
        output_dir.mkdir()
        output_path = output_dir / "frame_001.jpg"

        with pytest.raises(RuntimeError, match="Failed to save frame"):
            extractor.save_frame(frame, str(output_path))

class TestFramesExtractorResourceManagement:
    """Tests for FramesExtractor resource management."""

    def test_destructor_releases_resources(self, mocker, tmp_path):
        """Test __del__ properly releases VideoCapture resources."""
        test_video = tmp_path / "test.mp4"
        test_video.touch()

        mock_cap = mocker.Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mocker.patch('cv2.VideoCapture', return_value=mock_cap)

        extractor = FramesExtractor(str(test_video))

        # Call destructor explicitly
        extractor.__del__()

        # Verify release was called
        mock_cap.release.assert_called_once()
