from logging import getLogger, Logger
import logging
import torch
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import urllib.request
import cv2
import numpy as np
from core.model import DeepFakeDetector
from preprocessing.face_detector import FaceDetector
from preprocessing.face_extractor import FaceExtractor
from preprocessing.optimized_frames_extractor import create_frames_extractor
from torchvision import transforms
from utils.results_dataclasses import InferenceResult, VideoInferenceResult
from inference.score_aggregator import ScoreAggregator, AggregatedScore
from inference.batch_prefetcher import BatchPrefetcher
from utils.config_dataclasses import InferenceConfig
from config import PROJECT_ROOT, MODEL_FILENAME, FACE_DETECTION_MODEL_PATH, HF_MODEL_URL


class InferenceEngine:
    """Inference engine for VeridisQuo deepfake detection model."""

    CLASS_LABELS: List[str] = ["FAKE", "REAL"]
    """Supported class labels for model predictions (FAKE=0, REAL=1)"""
 
    def __init__(self, config: InferenceConfig, **kwargs):
        """Initialize InferenceEngine.
        Parameters:
            config (InferenceConfig): configuration for entire inference engine and model
            kwargs: override config parameters for critical settings, allowing to run without full config
                TODO
        Raises:
            ValueError: if config is not provided and no valid parameter is given for fallback
            RuntimeError: if model loading fails
        """
        super().__init__()

        self.logger: Logger = getLogger("/".join(__file__.split("/")[-2:]))
        """Logger instance for InferenceEngine"""

        if config is None:
            raise ValueError("Configuration is required. Please provide an InferenceConfig instance.")

        if not isinstance(config, InferenceConfig):
            raise TypeError(f"config must be InferenceConfig, got {type(config).__name__}")

        self.config: InferenceConfig = config

        #TODO: handle kwargs overrides for critical settings
        #Example: model_path, device, auto_download, etc.

        self.device: torch.device = self.config.device.get_device()
        """Device to run inference on (CPU or GPU)"""
        self.logger.info(f"Model will be loaded on device: {self.device} (cuda available: {torch.cuda.is_available()})")

        self.model_path: Path
        """Path to model weights file"""
        if self.config.paths.model_path is None:
            self.model_path = PROJECT_ROOT / "models" / "deepfake_detection" / MODEL_FILENAME
        else:
            self.model_path = Path(self.config.paths.model_path)

        if not self.model_path.exists() and self.config.inference_options.auto_download_model:
            self.logger.warning(f"Model not found at {self.model_path}, downloading from Hugging Face...")
            self._download_model()

        if not self.model_path.exists():
            error_msg = f"Model file not found at {self.model_path}. Set auto_download=True to download automatically."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Initializing VeridisQuo deepfake detection model...")

        self.model: DeepFakeDetector = DeepFakeDetector(config=self.config.model_architecture)
        """DeepFakeDetector model instance"""

        self._load_weights()
        self.logger.info("Model initialized and weights loaded successfully.")

        self.model = self.model.to(self.device)
        self.logger.info(f"Model moved to device: {self.device}")
        self.model.eval()
        self.logger.info("Model set to evaluation mode.")

        self.face_detector: Optional[FaceDetector] = None
        """FaceDetector instance for video inference, if available"""

        face_detection_model_path = Path(self.config.paths.face_detection_model_path) if self.config.paths.face_detection_model_path else FACE_DETECTION_MODEL_PATH

        if self.config.preprocessing.face_detection.enabled and face_detection_model_path.exists():
            self.logger.info("Initializing FaceDetector...")

            self.face_detector = FaceDetector(
                model_path=str(face_detection_model_path),
                device=str(self.device),
                min_face_size=self.config.preprocessing.face_detection.min_face_size,
                confidence_threshold=self.config.preprocessing.face_detection.confidence_threshold,
                only_keep_top=self.config.preprocessing.face_detection.only_keep_top
            )
        else:
            self.logger.warning(
                f"Face detection disabled or model not found at {face_detection_model_path}. "
                "Video inference will process entire frames instead of detected faces."
            )

        self.image_size: Tuple[int, int] = self.config.preprocessing.face_extraction.target_size
        """Image size for preprocessing (width, height)"""

        self.mean: List[float] = self.config.preprocessing.imagenet_mean
        """ImageNet mean for normalization"""

        self.std: List[float] = self.config.preprocessing.imagenet_std
        """ImageNet std for normalization"""

        self.padding: int = self.config.preprocessing.face_extraction.padding
        """Padding around detected face bounding box"""

        self.use_face_detection: bool = self.face_detector is not None
        """Whether face detection is available for video inference"""
        self.logger.info(f"Face detection enabled: {self.use_face_detection}")

        self.face_extractor: FaceExtractor = FaceExtractor(
            target_size=self.image_size,
            normalization=self.config.preprocessing.face_extraction.normalization,
            padding=self.padding
        )
        """Face extractor for consistent preprocessing"""

        self.transform: transforms.Compose = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        """Torchvision transform pipeline for normalization"""

        self.logger.info("InferenceEngine initialized successfully")

    def _download_model(self) -> None:
        """Download model weights from Hugging Face.
        Raises:
            RuntimeError: if download fails
        """
        try:
            self.logger.info(f"Downloading model from {HF_MODEL_URL}...")

            #create parent directory if needed
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

            def _progress_hook(count: int, block_size: int, total_size: int) -> None:
                """Progress callback for model download."""
                percent = int(count * block_size * 100 / total_size)
                if count % 100 == 0:
                    self.logger.info(f"Download progress: {percent}%")

            urllib.request.urlretrieve(
                HF_MODEL_URL,
                self.model_path,
                reporthook=_progress_hook
            )

            self.logger.info(f"Model downloaded successfully to {self.model_path}")

        except Exception as e:
            error_msg = f"Failed to download model from Hugging Face: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _load_weights(self) -> None:
        """Load pretrained weights into the model.
        Raises:
            RuntimeError: if weight loading fails
        """
        try:
            self.logger.info(f"Loading model weights from {self.model_path}...")

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            # Handle both direct state_dict and training checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                self.logger.debug("Loaded model_state_dict from training checkpoint")
            else:
                state_dict = checkpoint
                self.logger.debug("Loaded direct state_dict")

            self.model.load_state_dict(state_dict)

            self.logger.info("Model weights loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load model weights: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _extract_face(self,
        image: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Extract face region from image using detections.
        Parameters:
            image (np.ndarray): input image in BGR format [H, W, 3]
            detections (List[Dict]): face detections from face_detector
        Returns:
            np.ndarray: extracted and preprocessed face region [224, 224, 3]
        """
        if detections:
            #use the top face (highest confidence)
            top_face = detections[0]
            bbox = top_face['bbox']

            #extract face region with padding
            face_region, _ = self.face_extractor.extract_and_preprocess(
                image, bbox, normalize=False
            )
        else:
            #no face detected, use entire frame
            face_region = self.face_extractor.preprocess_face(image, normalize=False)

        return face_region

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image (wrapper around batch method).
        Parameters:
            image (np.ndarray): Input image in BGR format [H, W, 3]
        Returns:
            torch.Tensor: Preprocessed image tensor [1, 3, 224, 224]
        """
        return self._preprocess_image_batch([image])  #[1, 3, 224, 224]

    def _preprocess_image_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """Preprocess a batch of images efficiently with batched operations.
        Parameters:
            images (List[np.ndarray]): list of images in BGR format [H, W, 3]
        Returns:
            torch.Tensor: batched preprocessed tensor [N, 3, 224, 224]
        Raises:
            RuntimeError: if all images fail preprocessing
        """
        #batch face detection wrapped into a single YOLO forward pass
        if self.use_face_detection:
            detections_batch = self.face_detector.detect_faces_batch(images)
        else:
            detections_batch = [[] for _ in images]

        face_regions = []
        for idx, (image, detections) in enumerate(zip(images, detections_batch)):
            try:
                face_region = self._extract_face(image, detections)
                face_regions.append(face_region)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract face from image {idx}/{len(images)}: {e}, skipping frame")
                continue

        if not face_regions:
            raise RuntimeError("All images in batch failed face extraction")

        #batch transformations (BGR→RGB + ToTensor + Normalize)
        face_rgbs = [cv2.cvtColor(face, cv2.COLOR_BGR2RGB) for face in face_regions]
        tensors = [self.transform(face_rgb) for face_rgb in face_rgbs]

        #batch tensor [N, 3, 224, 224]
        batch_tensor = torch.stack(tensors, dim=0)

        return batch_tensor

    def _run_inference(self, preprocessed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run model inference on preprocessed tensor.
        Parameters:
            preprocessed (torch.Tensor): Preprocessed image tensor [N, 3, 224, 224]
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (logits, probabilities, prediction) all on GPU
        """
        with torch.no_grad():
            logits = self.model(preprocessed) #[N, 2]
            prediction = self.model.classifier.predict(logits) #[N]
            probabilities = self.model.classifier.predict_proba(logits) #[N, 2]
        return logits, probabilities, prediction

    def _predict_frame_batch(self, frames: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a batch of frames with batched operations.
        Parameters:
            frames (List[np.ndarray]): List of frame images in BGR format [H, W, 3]
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (logits, probabilities, prediction)
                - logits: [N, 2] on GPU
                - probabilities: [N, 2] on GPU
                - prediction: [N] on GPU (class indices)
        Raises:
            ValueError: if frames list is empty
            RuntimeError: if all frames fail preprocessing
        """
        if not frames:
            raise ValueError("frames list cannot be empty")

        #batch preprocessing (includes batched face detection + transforms)
        batch_tensor = self._preprocess_image_batch(frames)  # [N, 3, 224, 224] CPU

        #single GPU transfer for entire batch
        batch_tensor = batch_tensor.to(self.device)

        logits, probabilities, prediction = self._run_inference(batch_tensor)

        return logits, probabilities, prediction

    def _validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate that file (video or image) path exists.
        Parameters:
            file_path (Union[str, Path]): Path to video file
        Returns:
            Path: Validated path object
        Raises:
            FileNotFoundError: if file doesn't exist
        """
        file_path = Path(file_path)
        try:
            assert file_path.exists(), f"File not found: {file_path}"
            
        except AssertionError as e:
            error_msg = f"Invalid parameters: {e}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        return file_path

    def _build_frame_result(self,
        predicted_class: int,
        probs_cpu: np.ndarray,
        logits_cpu: np.ndarray,
        frame_idx: int
    ) -> InferenceResult:
        """Build InferenceResult for a single frame from batch predictions.
        Parameters:
            predicted_class (int): Predicted class index
            probs_cpu (np.ndarray): Probabilities array [N, 2]
            logits_cpu (np.ndarray): Logits array [N, 2]
            frame_idx (int): Index of frame in batch
        Returns:
            InferenceResult: Result object for the frame
        """
        confidence = float(probs_cpu[frame_idx, predicted_class])
        prediction_label = self.CLASS_LABELS[predicted_class]

        prob_dict = {
            label: float(probs_cpu[frame_idx, j])
            for j, label in enumerate(self.CLASS_LABELS)
        }

        return InferenceResult(
            prediction=prediction_label,
            confidence=confidence,
            probabilities=prob_dict,
            raw_logits=logits_cpu[frame_idx].tolist()
        )

    def _process_video_batches(self, prefetcher: BatchPrefetcher) -> List[InferenceResult]:
        """Process all batches from prefetcher and return frame results.
        Parameters:
            prefetcher (BatchPrefetcher): Initialized prefetcher with batches ready
        Returns:
            List[InferenceResult]: Results for all processed frames
        Raises:
            RuntimeError: if no frames were successfully processed
        """
        frame_results: List[InferenceResult] = []

        try:
            for batch_frames, batch_metadata in prefetcher:
                try:
                    logits, probabilities, predictions = self._predict_frame_batch(batch_frames)

                    #single CPU transfer per batch
                    probs_cpu = probabilities.cpu().numpy()  # [N, 2]
                    logits_cpu = logits.cpu().numpy()  # [N, 2]
                    predictions_cpu = predictions.cpu().numpy()  # [N]

                    #build results for each frame in batch
                    for i in range(len(batch_frames)):
                        predicted_class = int(predictions_cpu[i])
                        result = self._build_frame_result(predicted_class, probs_cpu, logits_cpu, i)
                        frame_results.append(result)

                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.logger.debug(
                                f"Frame {batch_metadata[i]['frame_number']} @ "
                                f"{batch_metadata[i]['timestamp']:.2f}s: "
                                f"{result.prediction} ({result.confidence:.4f})"
                            )

                except Exception as e:
                    self.logger.warning(f"Failed to process batch of {len(batch_frames)} frames: {e}")

        finally:
            prefetcher.stop()

        if not frame_results:
            raise RuntimeError("No frames were successfully processed")

        return frame_results

    def _aggregate_video_results(self,
        frame_results: List[InferenceResult],
        video_path: Path,
        aggregation_method: str
    ) -> VideoInferenceResult:
        """Aggregate frame results into video-level prediction.
        Parameters:
            frame_results (List[InferenceResult]): Results for all frames
            video_path (Path): Path to video file
            aggregation_method (str): Aggregation method to use
        Returns:
            VideoInferenceResult: Aggregated video prediction
        """
        self.logger.info(f"Processed {len(frame_results)} frames from video")

        aggregator = ScoreAggregator(method=aggregation_method)
        aggregated = aggregator.aggregate(frame_results)

        video_result = VideoInferenceResult(
            video_path=str(video_path),
            frame_results=frame_results,
            aggregate_prediction=aggregated.prediction,
            aggregate_confidence=aggregated.confidence,
            num_frames_analyzed=len(frame_results),
            aggregation_metadata={
                "method": aggregated.method,
                "frame_count": aggregated.frame_count,
                **aggregated.metadata
            }
        )

        self.logger.info(
            f"Video prediction: {aggregated.prediction} "
            f"(confidence: {aggregated.confidence:.4f}, "
            f"method: {aggregation_method}, "
            f"frames: {len(frame_results)})"
        )

        return video_result

    def predict_image(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference on a single image.
        Parameters:
            image (np.ndarray): Image array [H, W, 3] in BGR format
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (logits, probabilities, prediction)
                - logits: [1, 2] on GPU
                - probabilities: [1, 2] on GPU
                - prediction: [1] on GPU (class index)
        Raises:
            ValueError: if image is invalid
            RuntimeError: if inference fails
        """
        try:
            assert image is not None and image.size != 0, "Image is empty or None"

            preprocessed = self._preprocess_image(image)  #[1, 3, 224, 224]
            preprocessed = preprocessed.to(self.device)
            logits, probabilities, prediction = self._run_inference(preprocessed)
            return logits, probabilities, prediction

        except AssertionError as e:
            error_msg = f"Invalid parameter: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        except Exception as e:
            error_msg = f"Inference failed for image array: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def predict_image(self, image: Union[str, Path]) -> InferenceResult:
        """Run inference on a single image.
        Parameters:
            image (str or Path): Path to the image file
        Returns:
            InferenceResult: Prediction result with confidence and probabilities.
        Raises:
            FileNotFoundError: if image file doesn't exist
            RuntimeError: if inference fails
        """
        try:
            image_path = self._validate_file_path(image)
            image_array = cv2.imread(str(image_path))
            assert image_array is not None, f"Failed to load image: {image_path}"

            self.logger.info(f"Processing image: {image_path.name}")

            #reuse ndarray implementation
            logits, probabilities, prediction = self.predict_image(image_array)  # [1, 2], [1, 2], [1] on GPU

            #CPU transfer and result construction
            probs_cpu = probabilities.cpu().numpy()[0]  # [2]
            logits_cpu = logits.cpu().numpy()[0]  # [2]
            predicted_class = int(prediction.cpu().item())  # Use prediction instead of argmax

            confidence = float(probs_cpu[predicted_class])
            prediction_label = self.CLASS_LABELS[predicted_class]

            prob_dict = {
                label: float(prob)
                for label, prob in zip(self.CLASS_LABELS, probs_cpu)
            }

            result = InferenceResult(
                prediction=prediction_label,
                confidence=confidence,
                probabilities=prob_dict,
                raw_logits=logits_cpu.tolist()
            )

            self.logger.info(f"Predicted image: {prediction_label} (confidence: {confidence:.4f})")

            return result
        
        except Exception as e:
            error_msg = f"Inference failed for image array: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
    def predict_video(self,
        video_path: Union[str, Path],
        frames_per_second: Optional[int] =1,
        aggregation_method: str ="majority",
        use_pyav_extractor: bool =True,
        batch_size: int =8
    ) -> VideoInferenceResult:
        """Run inference on a video file.
        Parameters:
            video_path (str or Path): Path to the video file
            frames_per_second (int, optional): Number of frames to sample per second.
                If None, processes all frames (can be slow).
            aggregation_method (str): Method to aggregate frame predictions.
                Options: "majority", "average", "weighted_average", "max_confidence", "threshold"
            use_pyav_extractor (bool): Whether to use OptimizedFramesExtractor (PyAV)
            batch_size (int): Number of frames to process in a single batch (default: 8)
        Returns:
            VideoInferenceResult: Aggregated video prediction result
        Raises:
            FileNotFoundError: if video file doesn't exist
            RuntimeError: if inference fails
        """
        video_path = self._validate_file_path(video_path)
        self.logger.info(f"Processing video: {video_path.name}")

        try:
            #create frames extractor
            frames_extractor = create_frames_extractor(
                video_path=str(video_path),
                nb_fps=frames_per_second,
                device=str(self.device),
                use_optimized=use_pyav_extractor
            )

            self.logger.info(f"Extracting frames at {frames_per_second} FPS (video FPS: {frames_extractor.video_fps:.1f})")

            #create prefetcher for async batch preparation
            prefetcher = BatchPrefetcher(
                frames_extractor.extract_frames(),
                batch_size=batch_size,
                prefetch_size=2
            )
            prefetcher.start()

            frame_results = self._process_video_batches(prefetcher)

            #aggregate and build final result
            return self._aggregate_video_results(frame_results, video_path, aggregation_method)

        except Exception as e:
            error_msg = f"Video inference failed for {video_path}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def predict_video_batch(
        self,
        video_paths: List[Union[str, Path]],
        frames_per_second: Optional[int] = 1,
        aggregation_method: str = "majority",
        use_pyav_extractor: bool = True,
        batch_size: int = 8
    ) -> List[VideoInferenceResult]:
        """Run inference on a batch of videos.
        Parameters:
            video_paths (List[str or Path]): List of video file paths
            frames_per_second (int, optional): Number of frames to sample per second
            aggregation_method (str): Method to aggregate frame predictions
            use_pyav_extractor (bool): Whether to use OptimizedFramesExtractor
            batch_size (int): Number of frames to process in a single batch (default: 8)
        Returns:
            List[VideoInferenceResult]: List of video prediction results
        Raises:
            ValueError: if video_paths is empty
        """
        try:
            assert video_paths, "List of video paths cannot be empty"
            
        except AssertionError as e:
            error_msg = f"Invalid parameter: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info(f"Processing batch of {len(video_paths)} videos")

        results: List[VideoInferenceResult] = []

        for video_path in video_paths:
            try:
                result = self.predict_video(
                    video_path=video_path,
                    frames_per_second=frames_per_second,
                    aggregation_method=aggregation_method,
                    use_pyav_extractor=use_pyav_extractor,
                    batch_size=batch_size
                )
                results.append(result)

            except FileNotFoundError as e:
                self.logger.warning(f"Video file not found: {video_path}, skipping.")      
            
            except Exception as e:
                self.logger.error(f"Failed to process video {video_path}: {e}")
                
            finally:  
                #create error result
                error_result = VideoInferenceResult(
                    video_path=str(video_path),
                    frame_results=[],
                    aggregate_prediction="Inference Failed",
                    aggregate_confidence=0.0,
                    num_frames_analyzed=0,
                    aggregation_metadata={"error": str(e)}
                )
                results.append(error_result)
                continue

        self.logger.info(f"Batch video inference complete: {len(results)} results")
        return results