"""Pytest fixtures for frequal tests."""
import pytest
import numpy as np
import torch
import cv2


@pytest.fixture
def simple_grayscale_image():
    """Create a simple 224x224 grayscale image for testing.
    Returns:
        np.ndarray: grayscale image [224, 224]
    """
    image = np.zeros((224, 224), dtype=np.uint8)
    #add some patterns
    image[50:150, 50:150] = 255  #white square
    image[100:120, 100:120] = 128  #gray square inside
    return image


@pytest.fixture
def simple_rgb_image():
    """Create a simple 224x224 RGB image for testing.
    Returns:
        np.ndarray: RGB image [224, 224, 3] in BGR format
    """
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    #add some patterns
    image[50:150, 50:150, 0] = 255  #blue square
    image[100:120, 100:120, 1] = 255  #green square inside
    image[110:130, 110:130, 2] = 255  #red square inside
    return image


@pytest.fixture
def batch_rgb_images():
    """Create a batch of RGB images in numpy format for testing.
    Returns:
        np.ndarray: batch of RGB images [batch, 224, 224, 3] in BGR format
    """
    batch_size = 4
    images = np.zeros((batch_size, 224, 224, 3), dtype=np.uint8)
    for i in range(batch_size):
        #add different patterns for each image
        offset = i * 20
        images[i, 50+offset:150+offset, 50+offset:150+offset, 0] = 255
        images[i, 100+offset:120+offset, 100+offset:120+offset, 1] = 255
    return images


@pytest.fixture
def batch_torch_images():
    """Create a batch of RGB images in torch format for testing.
    Returns:
        torch.Tensor: batch of RGB images [batch, 3, 224, 224]
    """
    batch_size = 4
    images = torch.zeros((batch_size, 3, 224, 224))
    for i in range(batch_size):
        #add different patterns for each image
        offset = i * 20
        images[i, 0, 50+offset:150+offset, 50+offset:150+offset] = 1.0
        images[i, 1, 100+offset:120+offset, 100+offset:120+offset] = 1.0
    return images


@pytest.fixture
def checkerboard_image():
    """Create a checkerboard pattern image (high frequency content).
    Returns:
        np.ndarray: checkerboard image [224, 224, 3] in BGR format
    """
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 2):
        for j in range(0, 224, 2):
            if (i + j) % 4 == 0:
                image[i:i+2, j:j+2] = 255
    return image


@pytest.fixture
def smooth_gradient_image():
    """Create a smooth gradient image (low frequency content).
    Returns:
        np.ndarray: gradient image [224, 224, 3] in BGR format
    """
    x = np.linspace(0, 255, 224)
    y = np.linspace(0, 255, 224)
    xx, yy = np.meshgrid(x, y)
    image = np.stack([xx, yy, (xx + yy) / 2], axis=2).astype(np.uint8)
    return image


@pytest.fixture
def real_face_like_image():
    """Create a more realistic face-like image with various frequency components.
    Returns:
        np.ndarray: face-like image [224, 224, 3] in BGR format
    """
    image = np.ones((224, 224, 3), dtype=np.uint8) * 180  #skin tone base

    #add face oval
    center = (112, 112)
    axes = (80, 100)
    cv2.ellipse(image, center, axes, 0, 0, 360, (200, 180, 160), -1)

    #add eyes
    cv2.circle(image, (85, 95), 15, (50, 50, 50), -1)
    cv2.circle(image, (139, 95), 15, (50, 50, 50), -1)

    #add mouth
    cv2.ellipse(image, (112, 140), (30, 15), 0, 0, 180, (100, 80, 80), 2)

    #add some texture (high frequency)
    noise = np.random.randint(-10, 10, (224, 224, 3), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


@pytest.fixture
def small_image():
    """Create a small image (64x64) for edge case testing.
    Returns:
        np.ndarray: small image [64, 64, 3] in BGR format
    """
    image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return image


@pytest.fixture
def large_image():
    """Create a large image (512x512) for edge case testing.
    Returns:
        np.ndarray: large image [512, 512, 3] in BGR format
    """
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    return image
