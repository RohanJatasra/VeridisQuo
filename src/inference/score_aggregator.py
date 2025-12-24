from typing import List, Tuple, Dict, Any, Callable, Union
from dataclasses import dataclass
import logging

from inference.results_dataclasses import InferenceResult


@dataclass
class AggregatedScore:
    """Result of score aggregation across multiple frames.
    
    Attributes:
        prediction (str): Aggregated prediction label ('REAL' or 'FAKE')
        confidence (float): Aggregated confidence score (0-1)
        method (str): Aggregation method used
        frame_count (int): Number of frames aggregated
        metadata (Dict[str, Any]): Additional aggregation metadata
    """
    prediction: str
    confidence: float
    method: str
    frame_count: int
    metadata: Dict[str, Any]


class ScoreAggregator:
    """Aggregates frame-level predictions into video-level predictions.
    Supports multiple aggregation strategies:
    - majority: Majority vote of frame predictions
    - average: Average of confidence scores
    - weighted_average: Weighted average based on confidence
    - max_confidence: Prediction with highest confidence
    - threshold: Count frames above confidence threshold

    Example:
        >>> aggregator = ScoreAggregator(method="majority")
        >>> result = aggregator.aggregate(frame_results)
        >>> print(f"{result.prediction} (confidence: {result.confidence:.2%})")
    """

    AVAILABLE_METHODS: List[str] = [
        "majority",
        "average",
        "weighted_average",
        "max_confidence",
        "threshold"
    ]
    """Available aggregation methods"""

    def __init__(self,
        method: str = "majority",
        threshold: float = 0.5,
    ):
        """Initialize ScoreAggregator.
        Parameters:
            method (str): Aggregation method to use
                - "majority": majority vote of predictions
                - "average": average of confidence scores
                - "weighted_average": weighted average by confidence
                - "max_confidence": prediction with max confidence
                - "threshold": count frames above threshold
            threshold (float): Confidence threshold for "threshold" method
        Raises:
            ValueError: if method is not in AVAILABLE_METHODS
        """
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(
                f"Invalid aggregation method: {method}. "
                f"Available methods: {self.AVAILABLE_METHODS}"
            )

        self.method: str = method
        self.threshold: float = threshold
        self.logger: logging.Logger = logging.getLogger("/".join(__file__.split("/")[-2:]))

        self._aggregation_functions: Dict[str, Callable] = {
            "majority": self._aggregate_majority,
            "average": self._aggregate_average,
            "weighted_average": self._aggregate_weighted_average,
            "max_confidence": self._aggregate_max_confidence,
            "threshold": self._aggregate_threshold
        }

    def aggregate(self, frame_results: List[InferenceResult]) -> AggregatedScore:
        """Aggregate frame-level predictions into video-level prediction.
        Parameters:
            frame_results (List[InferenceResult]): List of frame predictions
        Returns:
            AggregatedScore: Aggregated prediction result
        Raises:
            ValueError: if frame_results is empty
        """
        if not frame_results:
            self.logger.error("No frame results provided for aggregation")
            raise ValueError("Cannot aggregate empty frame_results")

        self.logger.debug(f"Aggregating {len(frame_results)} frames using method '{self.method}'")

        aggregation_fn: callable = self._aggregation_functions[self.method]
        prediction: str
        confidence: float
        metadata: Dict[str, Any]
        prediction, confidence, metadata = aggregation_fn(frame_results)

        return AggregatedScore(
            prediction=prediction,
            confidence=confidence,
            method=self.method,
            frame_count=len(frame_results),
            metadata=metadata
        )
        
    def identify_labels(self, frame_results: List[InferenceResult]) -> set[str]:
        """Identify unique labels present in frame results. And assert that 
        all frames contain the same set of labels.
        Doing so allows dynamic handling of multi-class scenarios.
        Parameters:
            frame_results (List[InferenceResult]): Frame predictions
        Returns:
            List[str]: Unique labels found in predictions
        Raises:
            ValueError: if inconsistent labels are found across frames
        """
        unique_labels = set([label for label in frame_results[0].probabilities.keys()])
        
        for i, frame in enumerate(frame_results):
            frame_labels = frame.probabilities.keys()
            
            if frame_labels != unique_labels:
                err_msg = (
                    f"Inconsistent labels in frame results. "
                    f"Some frames have different labels: cannot aggregate. "
                    f"- Frame 0 labels: {unique_labels},\n- Frame {i} labels: {frame_labels}"
                )
                self.logger.error(err_msg)
                raise ValueError(err_msg)
            
        return unique_labels

    def _aggregate_majority(self, frame_results: List[InferenceResult]) -> Tuple[str, float, Dict[str, Any]]:
        """Majority vote aggregation. The most frequent prediction across frames becomes the final prediction.
        Confidence is the proportion of frames voting for that prediction. Supports multi-class scenarios.
        Parameters:
            frame_results (List[InferenceResult]): Frame predictions
        Returns:
            Tuple[str, float, Dict]: (prediction, confidence, metadata)
        """
        predictions = self.identify_labels(frame_results)
        
        counts: Dict[str, int] = {label: 0 for label in predictions}
        for result in frame_results:
            counts[result.prediction] += 1
        
        max_label = max(counts, key=counts.get)
        total = len(frame_results)
        confidence = counts[max_label] / total

        metadata = {}
        for label, count in counts.items():
            metadata[f"count_{label}"] = count
            metadata[f"ratio_{label}"] = count / total

        return max_label, confidence, metadata

    def _aggregate_average(self, 
        frame_results: List[InferenceResult], 
        threshold: float =0.5
    ) -> Union[Tuple[str, float, Dict[str, Any]], List[Tuple[str, float, Dict[str, Any]]]]:
        """Average confidence aggregation. Supports multi-class scenarios.
        Parameters:
            frame_results (List[InferenceResult]): Frame predictions
            threshold (float): Threshold to retain the prediction. 
        Returns:
            Tuple[str, float, Dict]: (prediction, confidence, metadata)
            List[Tuple[str, float, Dict]]: (prediction, confidence, metadata) for each label if multi-class
        """
        predictions = self.identify_labels(frame_results)
            
        avg_confidence_scores : Dict[str, float] = {}
        for label, scores in frame_results.probabilities.items():
            if scores:
                avg_confidence_scores[label] = sum(scores) / len(scores)
            else:
                avg_confidence_scores[label] = 0.0

        predictions_above_threshold = []
        confidences_above_threshold = []
        for label, avg_score in avg_confidence_scores.items():
            if avg_score > threshold:
                predictions_above_threshold.append(label)
                confidences_above_threshold.append(avg_score)
        
        metadata = {}
        for label, avg_score in avg_confidence_scores.items():
            metadata[f"avg_confidence_{label}"] = avg_score
        metadata[f"min_confidence"] = min(avg_confidence_scores.values())
        metadata[f"max_confidence"] = max(avg_confidence_scores.values())
        metadata[f"threshold_used"] = threshold

        if len(predictions_above_threshold) > 1:
            #multi-class scenario
            results = []
            for label, avg_score in zip(predictions_above_threshold, confidences_above_threshold):
                results.append((label, avg_score, metadata))
            return results
        
        elif len(predictions_above_threshold) == 1:
            #binary scenario
            return predictions_above_threshold[0], confidences_above_threshold[0], metadata
        
        else:
            self.logger.warning("No predictions above the specified threshold")
            return "UNKNOWN", 0.0, metadata

    def _aggregate_weighted_average(self, frame_results: List[InferenceResult]) -> Tuple[str, float, Dict[str, Any]]:
        """Weighted average aggregation. Supports multi-class scenarios.
        Parameters:
            frame_results (List[InferenceResult]): Frame predictions
        Returns:
            Tuple[str, float, Dict]: (prediction, confidence, metadata)
        """
        predictions = self.identify_labels(frame_results)
        
        weighted_sum = 0.0
        total_weight = 0.0

        counts: Dict[str, int] = {label: 0 for label in predictions}
        for result in frame_results:
            for label, prob in counts.items():
                counts[label] += 1
            
            fake_prob = result.probabilities["FAKE"]
            # Weight by the frame's confidence (max of REAL or FAKE probability)
            weight = result.confidence
            weighted_sum += fake_prob * weight
            total_weight += weight

        weighted_avg_fake = weighted_sum / total_weight if total_weight > 0 else 0.5

        if weighted_avg_fake > 0.5:
            prediction = "FAKE"
            confidence = weighted_avg_fake
        else:
            prediction = "REAL"
            confidence = 1 - weighted_avg_fake

        metadata = {
            "weighted_avg_fake": weighted_avg_fake,
            "weighted_avg_real": 1 - weighted_avg_fake,
            "total_weight": total_weight
        }

        return prediction, confidence, metadata

    def _aggregate_max_confidence(
        self,
        frame_results: List[InferenceResult]
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Maximum confidence aggregation.

        Selects the prediction from the frame with highest confidence.
        Useful when you trust the most confident prediction.

        Parameters:
            frame_results (List[InferenceResult]): Frame predictions

        Returns:
            Tuple[str, float, Dict]: (prediction, confidence, metadata)
        """
        # Find frame with maximum confidence
        max_result = max(frame_results, key=lambda r: r.confidence)

        prediction = max_result.prediction
        confidence = max_result.confidence

        metadata = {
            "max_confidence_frame": frame_results.index(max_result),
            "max_confidence_value": confidence,
            "max_frame_prediction": prediction
        }

        return prediction, confidence, metadata

    def _aggregate_threshold(
        self,
        frame_results: List[InferenceResult]
    ) -> Tuple[str, float, Dict[str, Any]]:
        """Threshold-based aggregation.

        Counts frames with FAKE probability above threshold.
        If majority of frames exceed threshold, predicts FAKE.

        Parameters:
            frame_results (List[InferenceResult]): Frame predictions

        Returns:
            Tuple[str, float, Dict]: (prediction, confidence, metadata)
        """
        above_threshold = 0
        total = len(frame_results)

        for result in frame_results:
            if result.probabilities["FAKE"] > self.threshold:
                above_threshold += 1

        ratio_above_threshold = above_threshold / total

        # If more than half the frames are above threshold, predict FAKE
        if ratio_above_threshold > 0.5:
            prediction = "FAKE"
            confidence = ratio_above_threshold
        else:
            prediction = "REAL"
            confidence = 1 - ratio_above_threshold

        metadata = {
            "threshold_used": self.threshold,
            "frames_above_threshold": above_threshold,
            "ratio_above_threshold": ratio_above_threshold,
            "frames_below_threshold": total - above_threshold
        }

        return prediction, confidence, metadata

    @staticmethod
    def compare_methods(
        frame_results: List[InferenceResult],
        methods: List[str] = None
    ) -> Dict[str, AggregatedScore]:
        """Compare multiple aggregation methods on the same frame results.

        Useful for analyzing which aggregation strategy works best.

        Parameters:
            frame_results (List[InferenceResult]): Frame predictions
            methods (List[str], optional): Methods to compare.
                If None, compares all available methods.

        Returns:
            Dict[str, AggregatedScore]: Results for each method

        Example:
            >>> results = ScoreAggregator.compare_methods(frame_results)
            >>> for method, score in results.items():
            ...     print(f"{method}: {score.prediction} ({score.confidence:.2%})")
        """
        if methods is None:
            methods = ScoreAggregator.AVAILABLE_METHODS

        comparison = {}
        for method in methods:
            aggregator = ScoreAggregator(method=method)
            comparison[method] = aggregator.aggregate(frame_results)

        return comparison
