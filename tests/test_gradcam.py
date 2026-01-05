"""
Tests for Grad-CAM module.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from gradcam import GradCAM, GradCAMVisualizer
from spatial.efficient_net_extractor import EfficientNetFeatureExtractor


class SimpleConvModel(nn.Module):
    """Simple model for testing Grad-CAM."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        features = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestGradCAM:
    """Test suite for GradCAM class."""

    def test_gradcam_initialization(self):
        """Test GradCAM initialization with hooks."""
        model = SimpleConvModel()
        target_layer = model.conv2

        gradcam = GradCAM(model, target_layer)

        assert gradcam.model is model
        assert gradcam.target_layer is target_layer
        assert gradcam.activations is None
        assert gradcam.gradients is None

    def test_gradcam_forward_hook(self):
        """Test that forward hook captures activations."""
        model = SimpleConvModel()
        gradcam = GradCAM(model, model.conv2)

        # Create dummy input
        x = torch.randn(2, 3, 224, 224)

        # Forward pass
        with torch.no_grad():
            _ = model(x)

        # Activations should be captured
        assert gradcam.activations is not None
        assert gradcam.activations.shape[0] == 2  # batch size
        assert gradcam.activations.shape[1] == 32  # channels

    def test_generate_cam_single_image(self):
        """Test CAM generation for single image."""
        model = SimpleConvModel()
        model.eval()
        gradcam = GradCAM(model, model.conv2)

        x = torch.randn(1, 3, 224, 224)
        cam = gradcam.generate_cam(x, target_class=0)

        # Check output shape
        assert cam.shape[0] == 1
        assert cam.dim() == 3  # [batch, H, W]

        # Check normalization
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0

    def test_generate_cam_batch(self):
        """Test CAM generation for batch of images."""
        model = SimpleConvModel()
        model.eval()
        gradcam = GradCAM(model, model.conv2)

        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        cam = gradcam.generate_cam(x, target_class=1)

        assert cam.shape[0] == batch_size

        # Each sample should be independently normalized
        for i in range(batch_size):
            assert cam[i].min() >= 0.0
            assert cam[i].max() <= 1.0

    def test_generate_cam_auto_target(self):
        """Test CAM generation with automatic target class selection."""
        model = SimpleConvModel()
        model.eval()
        gradcam = GradCAM(model, model.conv2)

        x = torch.randn(2, 3, 224, 224)
        cam = gradcam.generate_cam(x)  # No target_class specified

        assert cam.shape[0] == 2
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0

    def test_gradcam_call_with_upsample(self):
        """Test GradCAM call with upsampling."""
        model = SimpleConvModel()
        model.eval()
        gradcam = GradCAM(model, model.conv2)

        x = torch.randn(1, 3, 224, 224)
        cam = gradcam(x, target_class=0, upsample_to=(224, 224))

        # Check upsampled size
        assert cam.shape == torch.Size([1, 224, 224])
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0

    def test_gradcam_with_efficientnet(self):
        """Test GradCAM integration with EfficientNet."""
        # Create EfficientNet model
        model = EfficientNetFeatureExtractor(pretrained=False)

        # Target the last conv layer of EfficientNet
        target_layer = model.features[-1]

        gradcam = GradCAM(model, target_layer)

        # Create input
        x = torch.randn(1, 3, 224, 224)

        # Generate CAM (should work with tuple output from EfficientNet)
        cam = gradcam(x, upsample_to=(224, 224))

        assert cam.shape == torch.Size([1, 224, 224])
        assert cam.min() >= 0.0
        assert cam.max() <= 1.0


class TestGradCAMVisualizer:
    """Test suite for GradCAMVisualizer class."""

    def test_apply_colormap(self):
        """Test colormap application."""
        heatmap = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        colored = GradCAMVisualizer.apply_colormap(heatmap)

        assert colored.shape == (50, 50, 3)
        assert colored.dtype == np.uint8

    def test_overlay_heatmap(self):
        """Test heatmap overlay on image."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        heatmap = np.random.rand(100, 100).astype(np.float32)

        overlayed = GradCAMVisualizer.overlay_heatmap(image, heatmap, alpha=0.4)

        assert overlayed.shape == (100, 100, 3)
        assert overlayed.dtype == np.uint8

    def test_overlay_heatmap_resize(self):
        """Test heatmap overlay with size mismatch (automatic resize)."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(7, 7).astype(np.float32)

        overlayed = GradCAMVisualizer.overlay_heatmap(image, heatmap, alpha=0.5)

        assert overlayed.shape == (224, 224, 3)
        assert overlayed.dtype == np.uint8

    def test_tensor_to_numpy_3d(self):
        """Test tensor to numpy conversion for 3D tensors."""
        tensor = torch.randn(3, 224, 224)
        array = GradCAMVisualizer.tensor_to_numpy(tensor)

        # Should convert from [C, H, W] to [H, W, C]
        assert array.shape == (224, 224, 3)

    def test_tensor_to_numpy_2d(self):
        """Test tensor to numpy conversion for 2D tensors."""
        tensor = torch.randn(7, 7)
        array = GradCAMVisualizer.tensor_to_numpy(tensor)

        # Should remain [H, W]
        assert array.shape == (7, 7)

    def test_create_visualization(self):
        """Test complete visualization creation."""
        image = torch.randn(3, 224, 224)
        cam = torch.rand(224, 224)

        visualization = GradCAMVisualizer.create_visualization(
            image, cam, alpha=0.4, denormalize=True
        )

        assert visualization.shape == (224, 224, 3)
        assert visualization.dtype == np.uint8

    def test_create_visualization_no_denormalize(self):
        """Test visualization without denormalization."""
        # Image already in [0, 1] range
        image = torch.rand(3, 224, 224)
        cam = torch.rand(224, 224)

        visualization = GradCAMVisualizer.create_visualization(
            image, cam, alpha=0.3, denormalize=False
        )

        assert visualization.shape == (224, 224, 3)
        assert visualization.dtype == np.uint8


class TestGradCAMIntegration:
    """Integration tests for complete Grad-CAM pipeline."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from image to visualization."""
        # Create model
        model = SimpleConvModel()
        model.eval()

        # Setup Grad-CAM
        gradcam = GradCAM(model, model.conv2)

        # Create input
        x = torch.randn(1, 3, 224, 224)

        # Generate CAM
        cam = gradcam(x, target_class=0, upsample_to=(224, 224))

        # Create visualization
        visualization = GradCAMVisualizer.create_visualization(
            x[0], cam[0], alpha=0.4, denormalize=False
        )

        assert visualization.shape == (224, 224, 3)
        assert visualization.dtype == np.uint8
        assert visualization.min() >= 0
        assert visualization.max() <= 255

    def test_batch_processing(self):
        """Test processing batch of images."""
        model = SimpleConvModel()
        model.eval()
        gradcam = GradCAM(model, model.conv2)

        batch_size = 8
        x = torch.randn(batch_size, 3, 224, 224)

        # Generate CAMs for entire batch
        cams = gradcam(x, upsample_to=(224, 224))

        assert cams.shape == torch.Size([batch_size, 224, 224])

        # Create visualizations for each image
        visualizations = []
        for i in range(batch_size):
            vis = GradCAMVisualizer.create_visualization(
                x[i], cams[i], denormalize=False
            )
            visualizations.append(vis)

        assert len(visualizations) == batch_size
        for vis in visualizations:
            assert vis.shape == (224, 224, 3)

