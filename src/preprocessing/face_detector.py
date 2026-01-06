import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Union
from logging import getLogger, Logger
from ultralytics import YOLO
from pathlib import Path


class FaceDetector:
    """Detect faces in images using YOLO."""

    def __init__(self,
        model_path: str,
        min_face_size: int,
        confidence_threshold: float,
        only_keep_top: bool,
        device: Union[str, int]
    ):
        """Initializes the YOLO face detector.
        Parameters:
            model_path (Optional[str]): path to the YOLO face detection model.
                                       If None, uses the default path from config.
            min_face_size (int): minimum face size (in pixels) - faces smaller than this will be filtered out
            confidence_threshold (float): minimum confidence score to accept detection
            only_keep_top (bool): if True and multiple faces detected, keep only the highest confidence one
            device (Union[str, int]): device to run inference on. Can be 'cpu', 'cuda', or GPU index (0, 1, etc.)
        Raises:
            Exception: if YOLO model initialization fails or model file not found
        """
        self.min_face_size: int = min_face_size
        """Minimum face size (in pixels)"""
        self.confidence_threshold: float = confidence_threshold
        """Minimum confidence score to accept a detection"""
        self.only_keep_top: bool = only_keep_top
        """If True and multiple faces detected, keep only the highest confidence one"""
        self.device: Union[str, int] = device
        """Device to run inference on (cpu, cuda, or GPU index)"""
        self.logger: Logger = getLogger("/".join(__file__.split("/")[-2:]))
        """Logger for the FaceDetector class"""

        try:
            model_file = Path(model_path)
            
            assert model_file.exists(), f"YOLO model file does not exist at: {model_path}"

            self.detector: YOLO = YOLO(model_path)
            """YOLO face detector instance"""

            self.detector.to(device)

            self.logger.info(f"YOLO face detector initialized successfully from {model_path} on device: {device}")

        except AssertionError as e:
            err_msg = f"YOLO model initialization error: {e}"
            self.logger.fatal(err_msg)
            raise ValueError(err_msg)
        
        except Exception as e:
            self.logger.fatal(f"Failed to initialize YOLO face detector: {e}")
            raise

    def _process_single_result(self, result, return_landmarks: bool =False) -> List[Dict[str, Any]]:
        """Process a single YOLO result object into detection dictionaries.
        Parameters:
            result: YOLO result object for one image
            return_landmarks (bool): whether to include facial landmarks
        Returns:
            List[Dict[str, Any]]: list of detection dictionaries for this image
        """
        try:
            assert result and (not result.boxes or len(result.boxes) >= 0), "Invalid YOLO result object"
        
        except AssertionError as e:
            self.logger.error(f"Invalid YOLO result object: {e}")
            return []

        detections = []

        #process each detection
        for box in result.boxes:
            confidence = float(box.conf[0])

            #filter by confidence threshold
            if confidence < self.confidence_threshold:
                continue

            #get bounding box in xyxy format and convert to xywh
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            x, y = int(x1), int(y1)
            width, height = int(x2 - x1), int(y2 - y1)

            #filter by minimum face size
            if width < self.min_face_size or height < self.min_face_size:
                continue

            detection = {
                'bbox': [x, y, width, height],
                'confidence': confidence
            }

            if return_landmarks:
                detection['keypoints'] = {}

            detections.append(detection)

        if not detections:
            self.logger.debug("No valid detections after filtering by confidence and size")
            return []

        #sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        #keep only top detection if requested
        if len(detections) > 1 and self.only_keep_top:
            detections = [detections[0]]

        return detections

    def detect_faces_batch(self, images: List[np.ndarray], return_landmarks: bool =False) -> List[List[Dict[str, Any]]]:
        """Detect faces in multiple images with a single YOLO forward pass (batched).
        Parameters:
            images (List[np.ndarray]): list of images as numpy arrays (BGR format)
            return_landmarks (bool): whether to include facial landmarks in results
        Returns:
            List[List[Dict[str, Any]]]: list of detection lists (one per image)
        Raises:
            ValueError: if images list is empty or contains invalid images
        """
        try:
            assert images, "Images list cannot be empty"
            assert all(isinstance(img, np.ndarray) and img.size > 0 for img in images), "All images must be valid numpy arrays"
            
        except AssertionError as e:
            error_msg = f"Invalid images provided for batch face detection: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            #run YOLO inference on batch (single forward pass!)
            results = self.detector(images, verbose=False, device=self.device)

            if not results or len(results) == 0:
                self.logger.debug("No results from YOLO batch inference")
                return [[] for _ in images]

            #process each result
            all_detections = []
            for idx, result in enumerate(results):
                detections = self._process_single_result(result, return_landmarks)
                all_detections.append(detections)

                if detections:
                    self.logger.debug(f"Image {idx}: detected {len(detections)} face(s)")

            return all_detections

        except Exception as e:
            self.logger.error(f"Batch face detection failed: {e}")
            return [[] for _ in images]

    def detect_faces(self, image: np.ndarray, return_landmarks: bool =False) -> List[Dict[str, Any]]:
        """Detect faces in an image using YOLO.
        Parameters:
            image (np.ndarray): input image as numpy array (BGR format)
            return_landmarks (bool): whether to include facial landmarks in results (not supported with YOLO)
        Returns:
            List[Dict[str, Any]]: list of detection dictionaries, each containing:
            - bbox: bounding box as [x, y, width, height]
            - confidence: detection confidence score
            - keypoints: facial landmarks (empty dict if return_landmarks=True, as YOLO doesn't provide landmarks)
        Raises:
            ValueError: if image is invalid
        """
        #validate input image
        try:
            assert isinstance(image, np.ndarray), "Image must be a numpy array"
            assert image.size > 0, "Image array is empty"
            
        except AssertionError as e:
            error_msg = f"Invalid image provided for face detection: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            #run YOLO inference
            results = self.detector(image, verbose=False, device=self.device)

            if not results or len(results) == 0:
                self.logger.debug("No faces detected by YOLO")
                return []

            #process first result (single image) using helper method
            detections = self._process_single_result(results[0], return_landmarks)

            if detections:
                self.logger.debug(f"Detected {len(detections)} face(s)")
            else:
                self.logger.debug(
                    f"No faces with confidence >= {self.confidence_threshold} "
                    f"and size >= {self.min_face_size}px"
                )

            return detections

        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []

    def detect_single_face(self, image: np.ndarray, return_landmarks: bool =False) -> Optional[Dict[str, Any]]:
        """Detect a single face in an image (returns the one with the highest confidence score).
        Parameters:
            image (np.ndarray): input image as numpy array
            return_landmarks (bool): whether to include facial landmarks in result (not supported with YOLO)
        Returns:
            Optional[Dict[str, Any]]: None if no face found or dictionary containing:
            - bbox: bounding box as [x, y, width, height]
            - confidence: detection confidence score
            - keypoints: facial landmarks (empty dict if return_landmarks=True)
        Raises:
            ValueError: if provided image is invalid
        """
        detections = self.detect_faces(image, return_landmarks=return_landmarks)
        return None if not detections else detections[0]
