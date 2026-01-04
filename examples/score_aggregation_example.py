"""
Example: Using ScoreAggregator for video inference aggregation.

This example demonstrates how to use the ScoreAggregator to combine
frame-level predictions into video-level predictions using different
aggregation strategies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import ScoreAggregator, InferenceResult


def create_sample_frame_results():
    """Create sample frame results for demonstration."""
    # Simulating 20 frames with varying predictions
    frame_results = []

    # First 12 frames: mostly FAKE with high confidence
    for i in range(12):
        frame_results.append(InferenceResult(
            prediction="FAKE",
            confidence=0.75 + (i * 0.015),  # 0.75 to 0.915
            probabilities={
                "REAL": 0.25 - (i * 0.015),
                "FAKE": 0.75 + (i * 0.015)
            },
            raw_logits=[-0.5, 0.5]
        ))

    # Next 8 frames: REAL with moderate confidence
    for i in range(8):
        frame_results.append(InferenceResult(
            prediction="REAL",
            confidence=0.60 + (i * 0.01),  # 0.60 to 0.67
            probabilities={
                "REAL": 0.60 + (i * 0.01),
                "FAKE": 0.40 - (i * 0.01)
            },
            raw_logits=[0.3, -0.3]
        ))

    return frame_results


def demonstrate_all_methods():
    """Demonstrate all aggregation methods."""
    print("=" * 70)
    print("ScoreAggregator Demonstration")
    print("=" * 70)

    # Create sample data
    frame_results = create_sample_frame_results()

    print(f"\nDataset: {len(frame_results)} frames")
    fake_count = sum(1 for r in frame_results if r.prediction == "FAKE")
    real_count = sum(1 for r in frame_results if r.prediction == "REAL")
    print(f"  - FAKE frames: {fake_count}")
    print(f"  - REAL frames: {real_count}")

    avg_fake_conf = sum(r.probabilities["FAKE"] for r in frame_results) / len(frame_results)
    print(f"  - Average FAKE probability: {avg_fake_conf:.4f}")

    # Test each method
    print("\n" + "=" * 70)
    print("Results by Aggregation Method")
    print("=" * 70)

    methods_info = {
        "majority": "Most frequent prediction wins",
        "average": "Average of FAKE probabilities",
        "weighted_average": "Weighted by frame confidence",
        "max_confidence": "Frame with highest confidence decides",
        "threshold": "Count frames above threshold (0.5)"
    }

    for method in ScoreAggregator.AVAILABLE_METHODS:
        print(f"\n{method.upper()}")
        print(f"  Description: {methods_info[method]}")

        aggregator = ScoreAggregator(method=method, threshold=0.5)
        result = aggregator.aggregate(frame_results)

        print(f"  ➜ Prediction: {result.prediction}")
        print(f"  ➜ Confidence: {result.confidence:.4f} ({result.confidence*100:.2f}%)")
        print(f"  ➜ Metadata:")
        for key, value in result.metadata.items():
            if isinstance(value, float):
                print(f"      - {key}: {value:.4f}")
            else:
                print(f"      - {key}: {value}")


def compare_methods_side_by_side():
    """Compare all methods side by side."""
    print("\n" + "=" * 70)
    print("Side-by-Side Comparison")
    print("=" * 70)

    frame_results = create_sample_frame_results()

    # Use the built-in comparison method
    comparison = ScoreAggregator.compare_methods(frame_results)

    print(f"\n{'Method':<20} {'Prediction':<12} {'Confidence':<12}")
    print("-" * 44)

    for method, result in comparison.items():
        print(f"{method:<20} {result.prediction:<12} {result.confidence*100:>10.2f}%")


def demonstrate_custom_threshold():
    """Demonstrate threshold method with different thresholds."""
    print("\n" + "=" * 70)
    print("Threshold Method with Different Thresholds")
    print("=" * 70)

    frame_results = create_sample_frame_results()

    thresholds = [0.3, 0.5, 0.7, 0.9]

    print(f"\n{'Threshold':<12} {'Prediction':<12} {'Confidence':<12} {'Frames Above':<15}")
    print("-" * 51)

    for threshold in thresholds:
        aggregator = ScoreAggregator(method="threshold", threshold=threshold)
        result = aggregator.aggregate(frame_results)

        frames_above = result.metadata["frames_above_threshold"]
        total_frames = len(frame_results)

        print(
            f"{threshold:<12.1f} {result.prediction:<12} "
            f"{result.confidence*100:>10.2f}% {frames_above}/{total_frames:<15}"
        )


def demonstrate_edge_cases():
    """Demonstrate edge cases."""
    print("\n" + "=" * 70)
    print("Edge Cases")
    print("=" * 70)

    # Case 1: All FAKE
    print("\nCase 1: All frames predict FAKE")
    all_fake = [
        InferenceResult("FAKE", 0.9, {"REAL": 0.1, "FAKE": 0.9}, [])
        for _ in range(5)
    ]
    aggregator = ScoreAggregator(method="majority")
    result = aggregator.aggregate(all_fake)
    print(f"  ➜ {result.prediction} (confidence: {result.confidence:.2f})")

    # Case 2: Perfect 50/50 split
    print("\nCase 2: Perfect 50/50 split (REAL wins in majority vote)")
    split = [
        InferenceResult("FAKE", 0.8, {"REAL": 0.2, "FAKE": 0.8}, []),
        InferenceResult("FAKE", 0.7, {"REAL": 0.3, "FAKE": 0.7}, []),
        InferenceResult("REAL", 0.8, {"REAL": 0.8, "FAKE": 0.2}, []),
        InferenceResult("REAL", 0.7, {"REAL": 0.7, "FAKE": 0.3}, []),
    ]
    result = aggregator.aggregate(split)
    print(f"  ➜ {result.prediction} (confidence: {result.confidence:.2f})")

    # Case 3: One very confident frame vs many uncertain
    print("\nCase 3: One very confident FAKE vs three uncertain REAL")
    one_confident = [
        InferenceResult("FAKE", 0.99, {"REAL": 0.01, "FAKE": 0.99}, []),
        InferenceResult("REAL", 0.52, {"REAL": 0.52, "FAKE": 0.48}, []),
        InferenceResult("REAL", 0.51, {"REAL": 0.51, "FAKE": 0.49}, []),
        InferenceResult("REAL", 0.53, {"REAL": 0.53, "FAKE": 0.47}, []),
    ]

    # Compare methods for this case
    for method in ["majority", "average", "max_confidence"]:
        agg = ScoreAggregator(method=method)
        res = agg.aggregate(one_confident)
        print(f"  {method:20s} → {res.prediction} ({res.confidence:.2f})")


if __name__ == "__main__":
    demonstrate_all_methods()
    compare_methods_side_by_side()
    demonstrate_custom_threshold()
    demonstrate_edge_cases()

    print("\n" + "=" * 70)
    print("✅ Demonstration completed!")
    print("=" * 70)
