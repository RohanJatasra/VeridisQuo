import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
import logging
from mtcnn import MTCNN

class FaceDetector:
    """Detect faces in images using MTCNN."""

    def __init__(self,
        min_face_size: int =40,
        thresholds: List[float] = [0.6, 0.7, 0.7],
        scale_factor: float =0.709,
        confidence_threshold: float =0.9,
        only_keep_top: bool =True
    ):
        """Initializes the face detector.
        Parameters:
            min_face_size (int): minimum faces size (in pixels)
            thresholds (List[float]): MTCNN detection thresholds for three stages
            scale_factor (float): scale factor for image pyramid
            confidence_threshold (float): minimum confidence score to accept detection
            only_keep_top (bool): if True and multiple faces detected, keep only the highest confidence one
        Raises:
            Exception: if MTCNN initialization fails
        """
        self.min_face_size: int = min_face_size
        """Minimum faces size (in pixels)"""
        self.thresholds: List[float] = thresholds
        """MTCNN detection thresholds for three stages (P-Net, R-Net, O-Net)"""
        self.scale_factor: float = scale_factor
        """Scale factor for image pyramid decomposition"""
        self.confidence_threshold: float = confidence_threshold
        """Minimum confidence score to accept a detection"""
        self.only_keep_top: bool = only_keep_top
        """If True and multiple faces detected, keep only the highest confidence one"""
        self.logger: logging.Logger = logging.getLogger("preprocessing.FaceDetector")
        """Logger for the FaceDetector class"""

        try:
            self.detector: MTCNN = MTCNN(
                min_face_size=self.min_face_size,
                thresholds=self.thresholds,
                scale_factor=self.scale_factor
            )
            """MTCNN face detector instance"""
            self.logger.info("Initialized successfully")
            
        except Exception as e:
            self.logger.fatal(f"Failed to initialize MTCNN: {e}")
            raise

    def detect_faces(self,
        image: np.ndarray,
        return_landmarks: bool =False
    ) -> List[Dict[str, Any]]:
        """Detect faces in an image.
        Parameters:
            image (np.ndarray): input image as numpy array
            return_landmarks (bool): whether to include facial landmarks in results
        Returns:
            List[Dict[str, Any]]: list of detection dictionaries, each containing:
            - bbox: bounding box as [x, y, width, height]
            - confidence: detection confidence score
            - keypoints: facial landmarks (if return_landmarks=True)
        Raises:
            ValueError: if image is invalid
        """
        try:
            detected = self.detector.detect_faces(self._format_input_image(image))
            
        except Exception as e:
            self.logger.error(f"- detect_faces - face detection went wrong: {e}")
            return []

        if not detected:
            self.logger.debug("- detect_faces - no faces detected")
            return []

        filter_results = [f for f in detected if f['confidence'] >= self.confidence_threshold]

        if not filter_results:
            self.logger.debug(
                f"- detect_faces - no faces with confidence >= the provided threshold ({self.confidence_threshold})\n"
                f"- detect_faces - found faces: {len(detected)}"
            )
            return []

        filter_results.sort(key=lambda x: x['confidence'], reverse=True)
        if len(filter_results) > 1 and self.only_keep_top:
            self.logger.debug("- detect_faces - multiple faces detected, keeping the most confident one.")
            filter_results = [filter_results[0]]

        results = []
        for f in filter_results:
            result = {'bbox': f['box'], 'confidence': f['confidence']}

            if return_landmarks:
                result['keypoints'] = f['keypoints']

            results.append(result)

        self.logger.debug(f"- detect_faces - detected {len(results)} face(s)")
        return results

    def _force_rgb(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is in RGB format.
        Parameters:
            image (np.ndarray): input image as numpy array
        Returns:
            np.ndarray: image in RGB format
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _format_input_image(self, image: np.ndarray) -> np.ndarray:
        """Validate input image and ensure it's in RGB format.
        Parameters:
            image (np.ndarray): input image as numpy array
        Returns:
            np.ndarray: validated image in RGB format
        Raises:
            ValueError: if image is invalid
        """
        try:
            assert image is not None and isinstance(image, np.ndarray) and image.size > 0, f"invalid image: {image}"
            return self._force_rgb(image)
        
        except AssertionError as e:
            self.logger.error(f"- _format_input_image - {e}")
            raise ValueError(f"Invalid parameters provided for face detection: {e}")

    def detect_single_face(self,
        image: np.ndarray,
        return_landmarks: bool =False
    ) -> Optional[Dict[str, Any]]:
        """Detect a single face in an image (returns the one with the highest confidence score).
        Parameters:
            image (np.ndarray): input image as numpy array
            return_landmarks (bool): whether to include facial landmarks in result
        Returns:
            Optional[Dict[str, Any]]: None if no face found or dictionary containing:
            - bbox: bounding box as [x, y, width, height]
            - confidence: detection confidence score
            - keypoints: facial landmarks (if return_landmarks=True)
        Raises:
            ValueError: if provided image is invalid
        """
        detections = self.detect_faces(image, return_landmarks=return_landmarks)
        return None if not detections else detections[0] #should have previously been sorted by confidence
