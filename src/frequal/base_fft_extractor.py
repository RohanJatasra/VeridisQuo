import logging
from typing import Tuple
import torch.nn as nn
import math
from .base_frequency_extractor import BaseFrequencyExtractor


class BaseFFTExtractor(BaseFrequencyExtractor):
    """Base class for FFT-based frequency feature extractors.
    Provides common configuration and utilities for both NumPy and PyTorch implementations
    of FFT feature extraction for deepfake detection."""

    """Available channel processing modes"""
    AVAILABLE_WINDOW_FUNCTIONS = ["hann", "hamming", "blackman", "none", None]
    """Available window functions to reduce spectral leakage"""

    NUM_AZIMUTHAL_SECTORS = 8
    """Number of angular sectors for directional analysis (each covers 45 degrees)"""
    HIGH_FREQ_THRESHOLD_RATIO = 0.5
    """Ratio of max_radius defining high-frequency region (outer 50%)"""
    ARTIFACT_EDGE_WIDTH = 5
    """Width of edge regions for checkerboard artifact detection"""
    ARTIFACT_NUM_RADIAL_SAMPLES = 50
    """Number of radial samples for upsampling signature detection"""
    ARTIFACT_CENTER_REGION_SIZE = 20
    """Half-size of center region for high-frequency variance analysis"""
    DEFAULT_SMOOTHING_KERNEL_SIZE = 5
    """Default kernel size for spectrum smoothing"""
    ENERGY_BAND_START_MULTIPLIER = 2
    """Multiplier for extracting energy band start index"""
    ENERGY_BAND_END_MULTIPLIER = 3
    """Multiplier for extracting energy band end index"""
    EPSILON = 1e-10
    """Small constant for numerical stability in divisions"""

    def __init__(self,
        channel_mode: str,
        num_radial_bands: int,
        window_function: str,
        high_freq_emphasis: bool,
        feature_dim: int,
        logger_name: str
    ):
        """Initialize BaseFFTExtractor.
        Parameters:
            channel_mode (str): how to handle color channels
            num_radial_bands (int): number of radial frequency bands
            window_function (str): windowing function to reduce spectral leakage
            high_freq_emphasis (bool): emphasize high frequencies where deepfakes often have anomalies
            feature_dim (int): target output feature dimension
            logger_name (str): name for the logger instance
        Raises:
            ValueError: if parameters are invalid
        """
        super().__init__(
            channel_mode=channel_mode,
            feature_dim=feature_dim,
            logger_name=logger_name
        )

        try:
            assert num_radial_bands > 0, "Number of radial bands must be positive"
            assert window_function in self.AVAILABLE_WINDOW_FUNCTIONS, f"Invalid window function: {window_function}"

        except AssertionError as e:
            self.logger.fatal(f"Invalid parameters: {e}")
            raise ValueError(f"Invalid parameters: {e}")

        self.num_radial_bands: int = num_radial_bands
        """Number of radial frequency bands"""
        self.window_function: str = window_function
        """Window function type"""
        self.high_freq_emphasis: bool = high_freq_emphasis
        """Whether to emphasize high frequencies"""

        self._num_raw_features: int = self._calculate_raw_feature_dim()

        self.projection: nn.Linear = nn.Linear(self._num_raw_features, feature_dim)
        """Learnable projection from raw features to target dimension"""

    def _calculate_raw_feature_dim(self) -> int:
        """Calculate the number of raw features before projection, specifically for deepfake detection.
        Returns:
            int: number of raw features
        """
        # Radial bands: 8 bands × 4 statistics = 32
        # Radial profile: 8 values = 8
        # Azimuthal: 8 directions × 2 statistics = 16
        # Global spectral stats: 8
        # High-freq artifacts: 6
        # Cross-band interactions: 5
        return (self.num_radial_bands * 4) + self.num_radial_bands + 16 + 8 + 6 + 5

    @staticmethod
    def _compute_max_radius(center: Tuple[int, int]) -> float:
        """Compute maximum radius from center to corner.
        Parameters:
            center (Tuple[int, int]): center coordinates (cy, cx)
        Returns:
            float: maximum radius (distance to corner)
        """
        return math.sqrt(center[0]**2 + center[1]**2)

    def _get_sector_width(self) -> float:
        """Get angular width of each azimuthal sector.
        Returns:
            float: sector width in radians
        """
        return 2 * math.pi / self.NUM_AZIMUTHAL_SECTORS
