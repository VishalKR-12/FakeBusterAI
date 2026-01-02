"""
Face detection and cropping utilities for deepfake detection.
"""

import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import config


class FaceDetector:
    """Face detector using MTCNN."""
    
    def __init__(self, device=None):
        """
        Initialize face detector.
        
        Args:
            device: Device to run face detection on (default: auto-detect)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        self.mtcnn = MTCNN(
            image_size=config.FACE_SIZE,
            margin=0,
            min_face_size=config.MIN_FACE_SIZE,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=device
        )
    
    def detect_and_crop(self, image):
        """
        Detect face in image and return cropped face.
        
        Args:
            image: PIL Image or numpy array (RGB)
        
        Returns:
            cropped_face: PIL Image of cropped face (224x224) or None if no face detected
            bbox: Bounding box coordinates (x, y, width, height) or None
        """
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB if needed
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        try:
            # Detect and crop face
            face_tensor = self.mtcnn(image)
            
            if face_tensor is not None:
                # Convert tensor to PIL Image
                face_array = face_tensor.permute(1, 2, 0).cpu().numpy()
                face_array = (face_array * 255).astype(np.uint8)
                cropped_face = Image.fromarray(face_array)
                
                # Get bounding box (MTCNN doesn't directly return bbox, so we estimate)
                # For visualization purposes, we'll return the full image size
                bbox = (0, 0, image.width, image.height)
                
                return cropped_face, bbox
            else:
                return None, None
        except Exception as e:
            print(f"Face detection error: {e}")
            return None, None
    
    def detect_face_in_frame(self, frame):
        """
        Detect face in video frame.
        
        Args:
            frame: numpy array (BGR format from OpenCV)
        
        Returns:
            cropped_face: PIL Image or None
            bbox: Bounding box or None
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        return self.detect_and_crop(frame_pil)

