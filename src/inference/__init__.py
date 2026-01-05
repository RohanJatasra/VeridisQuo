"""
Inference module for VeridisQuo deepfake detection.

This module provides the InferenceEngine class and related utilities
for performing deepfake detection inference on images and videos.
"""

from .inference_engine import InferenceEngine
from .inference_results_dataclasses import InferenceResult, VideoInferenceResult
from .score_aggregator import ScoreAggregator, AggregatedScore
from .batch_prefetcher import BatchPrefetcher

__all__ = [
    "InferenceEngine",
    "InferenceResult",
    "VideoInferenceResult",
    "ScoreAggregator",
    "AggregatedScore",
    "BatchPrefetcher",
]
