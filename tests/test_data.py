"""
Tests for data loading and processing modules.
"""

import sys
import os
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.schemas import CSVItem, FrameCSVItem, FaceCSVItem
from dataset_builder.face_forensic import FaceForensicDataset
from utils.csv_services import CSVService
from config import TEST_CSV_FOLDER


class TestCSVService:
    """Tests for CSVService utility class."""

    def test_normalize_field_name(self):
        """Test field name normalization."""
        assert CSVService._normalize_field_name("File Path") == "File_Path"
        assert CSVService._normalize_field_name("File Size(MB)") == "File_Size_MB"
        assert CSVService._normalize_field_name("Frame_Number") == "Frame_Number"

    def test_row_csv_into_items(self):
        """Test converting CSV row to dataclass."""
        row = {
            "File_Path": "test/video.mp4",
            "Label": "FAKE",
            "Frame_Count": "100",
            "Width": "1920",
            "Height": "1080",
            "Codec": "h264",
            "File_Size_MB": "5.5"
        }

        item = CSVService.row_csv_into_items(row, CSVItem)

        assert isinstance(item, CSVItem)
        assert item.File_Path == "test/video.mp4"
        assert item.Label == "FAKE"
        assert item.Frame_Count == 100
        assert item.Width == 1920
        assert item.Height == 1080
        assert item.Codec == "h264"
        assert item.File_Size_MB == 5.5

    def test_row_csv_into_items_with_spaces(self):
        """Test converting CSV row with spaces in column names."""
        row = {
            "File Path": "test/video.mp4",
            "Label": "REAL",
            "Frame Count": "200",
            "File Size(MB)": "10.2"
        }

        item = CSVService.row_csv_into_items(row, CSVItem)

        assert item.File_Path == "test/video.mp4"
        assert item.Label == "REAL"
        assert item.Frame_Count == 200
        assert item.File_Size_MB == 10.2

    def test_row_csv_into_items_missing_fields(self):
        """Test handling missing fields with defaults."""
        row = {
            "File_Path": "test/video.mp4",
            "Label": "FAKE"
        }

        item = CSVService.row_csv_into_items(row, CSVItem)

        assert item.File_Path == "test/video.mp4"
        assert item.Label == "FAKE"
        assert item.Frame_Count == 0
        assert item.Width == 0
        assert item.Height == 0
        assert item.File_Size_MB == 0.0

    def test_save_and_load_csv(self):
        """Test saving and loading CSV with dataclasses."""
        items = [
            CSVItem(
                File_Path="video1.mp4",
                Label="FAKE",
                Frame_Count=100,
                Width=1920,
                Height=1080,
                Codec="h264",
                File_Size_MB=5.5
            ),
            CSVItem(
                File_Path="video2.mp4",
                Label="REAL",
                Frame_Count=200,
                Width=1280,
                Height=720,
                Codec="h265",
                File_Size_MB=3.2
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")

            # Save
            CSVService.save_csv(csv_path, items, CSVItem)

            # Load
            loaded_items = CSVService.load_csv(csv_path, CSVItem)

            assert len(loaded_items) == 2
            assert loaded_items[0].File_Path == "video1.mp4"
            assert loaded_items[0].Label == "FAKE"
            assert loaded_items[0].Frame_Count == 100
            assert loaded_items[1].File_Path == "video2.mp4"
            assert loaded_items[1].Label == "REAL"

    def test_save_empty_csv(self):
        """Test saving empty CSV creates headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "empty.csv")

            CSVService.save_csv(csv_path, [], CSVItem)

            assert os.path.exists(csv_path)

            # Check that headers are present
            loaded = CSVService.load_csv(csv_path, CSVItem)
            assert len(loaded) == 0

    def test_get_all_csvs(self):
        """Test getting all CSV files from folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some CSV files
            Path(tmpdir, "file1.csv").touch()
            Path(tmpdir, "file2.csv").touch()
            Path(tmpdir, "not_csv.txt").touch()

            csvs = CSVService.get_all_csvs(tmpdir)

            assert len(csvs) == 2
            assert "file1.csv" in csvs
            assert "file2.csv" in csvs
            assert "not_csv.txt" not in csvs

    def test_load_all_csvs(self):
        """Test loading all CSVs from a folder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            items1 = [CSVItem("video1.mp4", "FAKE", 100, 1920, 1080, "h264", 5.5)]
            items2 = [CSVItem("video2.mp4", "REAL", 200, 1280, 720, "h265", 3.2)]

            CSVService.save_csv(os.path.join(tmpdir, "dataset1.csv"), items1, CSVItem)
            CSVService.save_csv(os.path.join(tmpdir, "dataset2.csv"), items2, CSVItem)

            # Load all
            all_data = CSVService.load_all_csvs(tmpdir, CSVItem)

            assert len(all_data) == 2
            assert "dataset1" in all_data
            assert "dataset2" in all_data
            assert len(all_data["dataset1"]) == 1
            assert len(all_data["dataset2"]) == 1


class TestFaceForensicDataset:
    """Tests for FaceForensicDataset class."""

    def test_initialization(self):
        """Test FaceForensicDataset initialization."""
        if not os.path.exists(TEST_CSV_FOLDER):
            pytest.skip(f"Test CSV folder not found: {TEST_CSV_FOLDER}")

        dataset = FaceForensicDataset(TEST_CSV_FOLDER)

        assert dataset.csv_folder == TEST_CSV_FOLDER
        assert isinstance(dataset.items, dict)

    def test_load_all_csvs(self):
        """Test loading all CSV files from test dataset."""
        if not os.path.exists(TEST_CSV_FOLDER):
            pytest.skip(f"Test CSV folder not found: {TEST_CSV_FOLDER}")

        dataset = FaceForensicDataset(TEST_CSV_FOLDER)
        all_csv_data = dataset.load_all_csvs()

        assert isinstance(all_csv_data, dict), "All CSV data should be a dictionary"

        if len(all_csv_data) > 0:
            for key, value in all_csv_data.items():
                assert isinstance(key, str), "Dictionary keys should be strings"
                assert isinstance(value, list), "Dictionary values should be lists"
                assert all(isinstance(item, CSVItem) for item in value), \
                    "All items should be CSVItem instances"


class TestDataSchemas:
    """Tests for data schema dataclasses."""

    def test_csv_item_creation(self):
        """Test CSVItem dataclass creation."""
        item = CSVItem(
            File_Path="test.mp4",
            Label="FAKE",
            Frame_Count=100,
            Width=1920,
            Height=1080,
            Codec="h264",
            File_Size_MB=5.5
        )

        assert item.File_Path == "test.mp4"
        assert item.Label == "FAKE"
        assert item.Frame_Count == 100
        assert item.Width == 1920
        assert item.Height == 1080
        assert item.Codec == "h264"
        assert item.File_Size_MB == 5.5

    def test_frame_csv_item_creation(self):
        """Test FrameCSVItem dataclass creation."""
        item = FrameCSVItem(
            Frame_Path="frame_001.jpg",
            Video_Path="test.mp4",
            Label="FAKE",
            Width=1920,
            Height=1080,
            Frame_Number=1,
            Timestamp=0.033,
            Dataset="DeepFakeDetection"
        )

        assert item.Frame_Path == "frame_001.jpg"
        assert item.Video_Path == "test.mp4"
        assert item.Label == "FAKE"
        assert item.Frame_Number == 1
        assert item.Timestamp == 0.033

    def test_face_csv_item_creation(self):
        """Test FaceCSVItem dataclass creation."""
        item = FaceCSVItem(
            Face_Path="face_001_00.jpg",
            Video_Path="test.mp4",
            Label="FAKE",
            Frame_Number=1,
            Dataset="DeepFakeDetection",
            Confidence=0.99,
            BBox_X=100,
            BBox_Y=150,
            BBox_Width=200,
            BBox_Height=250,
            Face_Width=224,
            Face_Height=224
        )

        assert item.Face_Path == "face_001_00.jpg"
        assert item.Confidence == 0.99
        assert item.BBox_X == 100
        assert item.BBox_Y == 150
        assert item.Face_Width == 224
        assert item.Face_Height == 224


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
