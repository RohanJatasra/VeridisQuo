from logging import getLogger, Logger
import torch
from pathlib import Path
from typing import Union, List, Optional, Tuple
import urllib.request
import cv2
import numpy as np

from core.model import DeepFakeDetector
from preprocessing.face_detector import FaceDetector
from preprocessing.face_extractor import FaceExtractor
from preprocessing.optimized_frames_extractor import create_frames_extractor
from config import (
    PROJECT_ROOT, FACE_DETECTION_MODEL_PATH, HF_MODEL_URL, MODEL_FILENAME,
    IMAGE_SIZE, MEAN, STD, MIN_FACE_SIZE, FACE_DETECT_CONFIDENCE_THRESHOLD,
    ONLY_KEEP_TOP_FACE, FACE_EXTRACT_PADDING
)
from torchvision import transforms
from inference.results_dataclasses import InferenceResult, VideoInferenceResult
from inference.score_aggregator import ScoreAggregator, AggregatedScore


class InferenceEngine:
    """Inference engine for VeridisQuo deepfake detection model.
    Provides a complete pipeline for deepfake detection including:
    - Automatic model downloading from Hugging Face
    - Image preprocessing (resize, normalization)
    - Inference on single images, batches, and videos
    - Face detection integration for video analysis
    - CPU/GPU support
    """

    CLASS_LABELS: List[str] = ["REAL", "FAKE"]
    """Class labels for model predictions"""
 
    def __init__(self,
        model_path: Optional[Union[str, Path]] =None,
        device: Optional[Union[str, torch.device]] =None,
        auto_download: bool =True
    ):
        """Initialize InferenceEngine.
        Parameters:
            model_path (str or Path, optional): Path to model weights (.pth file).
                If None and auto_download=True, downloads from Hugging Face.
            device (str or torch.device, optional): Device to run inference on.
                If None, automatically selects CUDA if available, else CPU.
            auto_download (bool): Whether to automatically download model from HuggingFace
                if model_path is None or doesn't exist.
        Raises:
            ValueError: if parameters are invalid
            RuntimeError: if model loading fails
        """
        super().__init__()

        self.logger: Logger = getLogger("/".join(__file__.split("/")[-2:]))
        """Logger instance for InferenceEngine"""
        
        self.device: torch.device
        """Device to run inference on (CPU or GPU)"""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.logger.info(f"Model will be loaded on device: {self.device} (cuda available: {torch.cuda.is_available()})")

        #does not download model if not needed
        self.model_path: Path
        """Path to model weights file"""
        if model_path is None:
            self.model_path: Path = PROJECT_ROOT / "models" / "deepfake_detector" / MODEL_FILENAME
        else:
            self.model_path: Path = Path(model_path)

        if not self.model_path.exists() and auto_download:
            self.logger.warning(f"Model not found at {model_path}, downloading from Hugging Face...")
            self._download_model()

        if not self.model_path.exists():
            error_msg = f"Model file not found at {self.model_path}. Set auto_download=True to download automatically."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Initializing VeridisQuo deepfake detection model...")
        use_gpu: bool = str(self.device).startswith('cuda')
        
        self.model: DeepFakeDetector = DeepFakeDetector(pretrained=True, use_gpu=use_gpu)
        """DeepFakeDetector model instance"""
        
        self._load_weights()
        self.logger.info("Model initialized and weights loaded successfully.")
        
        self.model = self.model.to(self.device)
        self.logger.info(f"Model moved to device: {self.device}")
        self.model.eval()
        self.logger.info("Model set to evaluation mode.")

        self.face_detector: Optional[FaceDetector] = None
        """FaceDetector instance for video inference, if available"""

        if FACE_DETECTION_MODEL_PATH.exists():
            self.logger.info("Initializing FaceDetector...")
            
            self.face_detector = FaceDetector(
                model_path=str(FACE_DETECTION_MODEL_PATH),
                device=str(self.device),
                min_face_size=MIN_FACE_SIZE,
                confidence_threshold=FACE_DETECT_CONFIDENCE_THRESHOLD,
                only_keep_top=ONLY_KEEP_TOP_FACE
            )
        else:
            self.logger.warning(
                f"Face detection model not found at {FACE_DETECTION_MODEL_PATH}. "
                "Video inference will process entire frames instead of detected faces."
            )

        self.image_size: Tuple[int, int] = IMAGE_SIZE
        """Image size for preprocessing (width, height)"""

        self.mean: List[float] = MEAN
        """ImageNet mean for normalization"""

        self.std: List[float] = STD
        """ImageNet std for normalization"""

        self.padding: int = FACE_EXTRACT_PADDING
        """Padding around detected face bounding box"""

        self.use_face_detection: bool = self.face_detector is not None
        """Whether face detection is available for video inference"""
        self.logger.info(f"Face detection enabled: {self.use_face_detection}")

        self.face_extractor: FaceExtractor = FaceExtractor(
            target_size=self.image_size,
            normalization=None,  # Pas de normalisation ici, gérée par torchvision
            padding=self.padding
        )
        """Face extractor for consistent preprocessing"""

        #pipeline de normalisation torchvision (optimisé, support CPU/GPU)
        self.transform: transforms.Compose = transforms.Compose([
            transforms.ToTensor(),  # [0, 255] uint8 → [0, 1] float32, [H,W,C] → [C,H,W]
            transforms.Normalize(mean=self.mean, std=self.std)  #ImageNet normalization
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
            
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            self.logger.info("Model weights loaded successfully")

        except Exception as e:
            error_msg = f"Failed to load model weights: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image for model input.
        Parameters:
            image (np.ndarray): Input image in BGR format [H, W, 3]
        Returns:
            torch.Tensor: Preprocessed image tensor [1, 3, 224, 224]
        Raises:
            ValueError: if image is invalid
        """
        try:
            assert image is not None and image.size != 0, "Image is empty or None"
        
            #if face detection is available, detect and extract face
            if self.use_face_detection:
                detections = self.face_detector.detect_faces(image, return_landmarks=False)

                if detections:
                    #use the top face (highest confidence or largest area)
                    top_face = detections[0]
                    bbox = top_face['bbox']

                    #extract face region with padding
                    face_region, _ = self.face_extractor.extract_and_preprocess(
                        image, bbox, normalize=False
                    )
                    self.logger.debug(f"Face detected at bbox: {bbox}")
                    
                else:
                    #no face detected, resize entire image
                    err_msg = "No face detected"
                    self.logger.warning(err_msg)
                    raise RuntimeWarning(err_msg) #falling to except block
            
            else:
                err_msg = "Face detection not available"
                self.logger.warning(err_msg)
                raise RuntimeWarning(err_msg) #falling to except block
            
        except AssertionError as e:
            err_msg = "Invalid image: empty or None"
            self.logger.error(err_msg)
            raise ValueError(err_msg)
        
        except RuntimeWarning as w:
            self.logger.warning(f"Face detection failed: {w}. Using entire frame.")
            face_region = self.face_extractor.preprocess_face(image, normalize=False)
            
        #BGR to RGB for Torchvision
        face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

        #torchvision transforms => ToTensor + ImageNet normalization
        preprocessed = self.transform(face_rgb)  #[3, 224, 224]

        preprocessed = preprocessed.unsqueeze(0)  #[1, 3, 224, 224] (add batch dim)

        return preprocessed

    def predict_image(self, image_path: Union[str, Path]) -> InferenceResult:
        """Run inference on a single image.
        Parameters:
            image_path (str or Path): Path to the image file
        Returns:
            InferenceResult: Prediction result with label, confidence, probabilities
        Raises:
            FileNotFoundError: if image file doesn't exist
            ValueError: if image is invalid
            RuntimeError: if inference fails
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Load image with OpenCV (BGR format)
            image = cv2.imread(str(image_path))

            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            self.logger.info(f"Processing image: {image_path.name}")

            # Preprocess image
            preprocessed = self._preprocess_image(image)  # [1, 3, 224, 224]
            preprocessed = preprocessed.to(self.device)

            # Run inference
            with torch.no_grad():
                logits = self.model(preprocessed)  # [1, 2]
                probabilities = torch.softmax(logits, dim=1)  # [1, 2]

            # Extract results
            probs_cpu = probabilities.cpu().numpy()[0]  # [2]
            logits_cpu = logits.cpu().numpy()[0]  # [2]

            predicted_class = int(probs_cpu.argmax())
            confidence = float(probs_cpu[predicted_class])
            prediction_label = self.CLASS_LABELS[predicted_class]

            # Build probability dictionary
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

            self.logger.info(
                f"Prediction: {prediction_label} (confidence: {confidence:.4f})"
            )

            return result

        except Exception as e:
            error_msg = f"Inference failed for image {image_path}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def predict_image_batch(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 8
    ) -> List[InferenceResult]:
        """Run inference on a batch of images.
        Parameters:
            image_paths (List[str or Path]): List of image file paths
            batch_size (int): Batch size for processing (default: 8)
        Returns:
            List[InferenceResult]: List of prediction results
        Raises:
            ValueError: if image_paths is empty
            RuntimeError: if inference fails
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")

        self.logger.info(f"Processing batch of {len(image_paths)} images")

        results = []

        # Process in batches for memory efficiency
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            valid_indices = []

            # Load and preprocess images
            for idx, img_path in enumerate(batch_paths):
                try:
                    image = cv2.imread(str(img_path))

                    if image is None:
                        self.logger.warning(f"Failed to load image: {img_path}, skipping")
                        results.append(InferenceResult(
                            prediction="ERROR",
                            confidence=0.0,
                            probabilities={"REAL": 0.0, "FAKE": 0.0},
                            raw_logits=[0.0, 0.0]
                        ))
                        continue

                    preprocessed = self._preprocess_image(image)  # [1, 3, 224, 224]
                    batch_tensors.append(preprocessed)
                    valid_indices.append(len(results))

                except Exception as e:
                    self.logger.warning(f"Error preprocessing {img_path}: {e}, skipping")
                    results.append(InferenceResult(
                        prediction="ERROR",
                        confidence=0.0,
                        probabilities={"REAL": 0.0, "FAKE": 0.0},
                        raw_logits=[0.0, 0.0]
                    ))
                    continue

            if not batch_tensors:
                continue

            # Concatenate batch
            batch_tensor = torch.cat(batch_tensors, dim=0)  # [N, 3, 224, 224]
            batch_tensor = batch_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                logits = self.model(batch_tensor)  # [N, 2]
                probabilities = torch.softmax(logits, dim=1)  # [N, 2]

            # Extract results
            probs_cpu = probabilities.cpu().numpy()  # [N, 2]
            logits_cpu = logits.cpu().numpy()  # [N, 2]

            for j in range(len(batch_tensors)):
                probs = probs_cpu[j]
                logit = logits_cpu[j]

                predicted_class = int(probs.argmax())
                confidence = float(probs[predicted_class])
                prediction_label = self.CLASS_LABELS[predicted_class]

                prob_dict = {
                    label: float(prob)
                    for label, prob in zip(self.CLASS_LABELS, probs)
                }

                result = InferenceResult(
                    prediction=prediction_label,
                    confidence=confidence,
                    probabilities=prob_dict,
                    raw_logits=logit.tolist()
                )

                # Insert at the correct position
                if valid_indices[j] == len(results):
                    results.append(result)
                else:
                    results.insert(valid_indices[j], result)

        self.logger.info(f"Batch inference complete: {len(results)} results")
        return results

    def predict_video(
        self,
        video_path: Union[str, Path],
        frames_per_second: Optional[int] = 1,
        aggregation_method: str = "majority",
        use_optimized_extractor: bool = True
    ) -> VideoInferenceResult:
        """Run inference on a video file.
        Parameters:
            video_path (str or Path): Path to the video file
            frames_per_second (int, optional): Number of frames to sample per second.
                If None, processes all frames (can be slow).
            aggregation_method (str): Method to aggregate frame predictions.
                Options: "majority", "average", "weighted_average", "max_confidence", "threshold"
            use_optimized_extractor (bool): Whether to use OptimizedFramesExtractor (PyAV)
        Returns:
            VideoInferenceResult: Aggregated video prediction result
        Raises:
            FileNotFoundError: if video file doesn't exist
            RuntimeError: if inference fails
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.logger.info(f"Processing video: {video_path.name}")

        try:
            # Create frames extractor
            frames_extractor = create_frames_extractor(
                video_path=str(video_path),
                nb_fps=frames_per_second,
                device=str(self.device),
                use_optimized=use_optimized_extractor
            )

            self.logger.info(
                f"Extracting frames at {frames_per_second} FPS "
                f"(video FPS: {frames_extractor.video_fps:.1f})"
            )

            # Process frames
            frame_results = []
            frame_count = 0

            for frame, metadata in frames_extractor.extract_frames():
                frame_count += 1

                try:
                    # Preprocess frame
                    preprocessed = self._preprocess_image(frame)  # [1, 3, 224, 224]
                    preprocessed = preprocessed.to(self.device)

                    # Run inference
                    with torch.no_grad():
                        logits = self.model(preprocessed)  # [1, 2]
                        probabilities = torch.softmax(logits, dim=1)  # [1, 2]

                    # Extract results
                    probs_cpu = probabilities.cpu().numpy()[0]  # [2]
                    logits_cpu = logits.cpu().numpy()[0]  # [2]

                    predicted_class = int(probs_cpu.argmax())
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

                    frame_results.append(result)

                    self.logger.debug(
                        f"Frame {metadata['frame_number']} @ {metadata['timestamp']:.2f}s: "
                        f"{prediction_label} ({confidence:.4f})"
                    )

                except Exception as e:
                    self.logger.warning(f"Failed to process frame {frame_count}: {e}, skipping")
                    continue

            if not frame_results:
                raise RuntimeError("No frames were successfully processed")

            self.logger.info(f"Processed {len(frame_results)} frames from video")

            # Aggregate frame predictions
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

        except Exception as e:
            error_msg = f"Video inference failed for {video_path}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def predict_video_batch(
        self,
        video_paths: List[Union[str, Path]],
        frames_per_second: Optional[int] = 1,
        aggregation_method: str = "majority",
        use_optimized_extractor: bool = True
    ) -> List[VideoInferenceResult]:
        """Run inference on a batch of videos.
        Parameters:
            video_paths (List[str or Path]): List of video file paths
            frames_per_second (int, optional): Number of frames to sample per second
            aggregation_method (str): Method to aggregate frame predictions
            use_optimized_extractor (bool): Whether to use OptimizedFramesExtractor
        Returns:
            List[VideoInferenceResult]: List of video prediction results
        Raises:
            ValueError: if video_paths is empty
        """
        if not video_paths:
            raise ValueError("video_paths cannot be empty")

        self.logger.info(f"Processing batch of {len(video_paths)} videos")

        results = []

        for video_path in video_paths:
            try:
                result = self.predict_video(
                    video_path=video_path,
                    frames_per_second=frames_per_second,
                    aggregation_method=aggregation_method,
                    use_optimized_extractor=use_optimized_extractor
                )
                results.append(result)

            except Exception as e:
                self.logger.error(f"Failed to process video {video_path}: {e}")
                # Create error result
                error_result = VideoInferenceResult(
                    video_path=str(video_path),
                    frame_results=[],
                    aggregate_prediction="ERROR",
                    aggregate_confidence=0.0,
                    num_frames_analyzed=0,
                    aggregation_metadata={"error": str(e)}
                )
                results.append(error_result)

        self.logger.info(f"Batch video inference complete: {len(results)} results")
        return results