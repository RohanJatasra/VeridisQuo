"""
Pytest fixtures for inference module tests.

This module provides shared fixtures for testing the InferenceEngine
and related components.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_model():
    """Create a mock DeepFakeDetector model.

    Returns:
        MagicMock: Mock model that returns fake logits
    """
    model = MagicMock()
    # Return logits favoring FAKE class
    model.return_value = torch.tensor([[1.0, 2.0]])
    model.parameters.return_value = [torch.randn(10, 10)]
    return model


@pytest.fixture
def sample_image_pil():
    """Create a sample PIL Image for testing.

    Returns:
        PIL.Image: RGB image (224, 224)
    """
    return Image.new('RGB', (224, 224), color='red')


@pytest.fixture
def sample_image_numpy():
    """Create a sample numpy array image for testing.

    Returns:
        np.ndarray: RGB image array (224, 224, 3)
    """
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_batch_images():
    """Create a batch of sample images for testing.

    Returns:
        List[PIL.Image]: List of 3 RGB images
    """
    return [
        Image.new('RGB', (224, 224), color='red'),
        Image.new('RGB', (224, 224), color='blue'),
        Image.new('RGB', (224, 224), color='green')
    ]


@pytest.fixture
def sample_logits_fake():
    """Create sample logits favoring FAKE class.

    Returns:
        torch.Tensor: Logits tensor (1, 2) with FAKE class having higher value
    """
    return torch.tensor([[1.0, 3.0]])


@pytest.fixture
def sample_logits_real():
    """Create sample logits favoring REAL class.

    Returns:
        torch.Tensor: Logits tensor (1, 2) with REAL class having higher value
    """
    return torch.tensor([[3.0, 1.0]])


@pytest.fixture
def mock_inference_engine(mock_model):
    """Create a mock InferenceEngine for testing.

    Parameters:
        mock_model: Mock model fixture

    Returns:
        InferenceEngine: Mock engine with patched dependencies
    """
    with patch('inference.inference_engine.DeepFakeDetector') as mock_detector, \
         patch('inference.inference_engine.Path.exists') as mock_exists, \
         patch('torch.load') as mock_load:

        mock_exists.return_value = True
        mock_detector.return_value = mock_model
        mock_load.return_value = {}

        from inference import InferenceEngine

        return InferenceEngine(
            model_path="test_model.pth",
            auto_download=False,
            use_face_detection=False
        )
