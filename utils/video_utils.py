"""
Video processing utilities for frame extraction and analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import config


def extract_frames(video_path: str, fps: float = None) -> List[Tuple[int, np.ndarray]]:
    """
    Extract frames from video at specified FPS.
    
    Args:
        video_path: Path to video file
        fps: Frames per second to extract (default: config.FPS)
    
    Returns:
        List of tuples (frame_number, frame_array)
    """
    if fps is None:
        fps = config.FPS
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frames.append((frame_count, frame))
        
        frame_count += 1
    
    cap.release()
    return frames


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return info


def resize_frame(frame: np.ndarray, max_size: int = 800) -> np.ndarray:
    """
    Resize frame for display while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        max_size: Maximum dimension size
    
    Returns:
        Resized frame
    """
    height, width = frame.shape[:2]
    
    if max(height, width) <= max_size:
        return frame
    
    if height > width:
        new_height = max_size
        new_width = int(width * (max_size / height))
    else:
        new_width = max_size
        new_height = int(height * (max_size / width))
    
    return cv2.resize(frame, (new_width, new_height))

