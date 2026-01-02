"""
Preprocessing script for DeepFake Detection Pipeline
Extracts frames from videos, detects faces, and splits data into train/val/test sets.
Ensures no frame leakage by splitting at video level.
"""

import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import shutil
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import config

# Set random seeds for reproducibility
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

# Determine device (check CUDA availability)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize face detector
if config.FACE_DETECTION_METHOD.lower() == "mtcnn":
    mtcnn = MTCNN(
        image_size=config.FACE_SIZE,
        margin=0,
        min_face_size=config.MIN_FACE_SIZE,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        device=device
    )
else:
    raise ValueError(f"Face detection method '{config.FACE_DETECTION_METHOD}' not implemented. Use 'mtcnn'.")


def extract_frames(video_path, output_dir, label, video_name):
    """
    Extract frames from a video at specified FPS and detect faces.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted face crops
        label: 'real' or 'fake'
        video_name: Name of the video (without extension)
    
    Returns:
        Number of frames successfully extracted
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / config.FPS))  # Calculate frame interval
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified FPS
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Detect face
            try:
                face_tensor = mtcnn(frame_pil)
                if face_tensor is not None:
                    # Convert tensor to numpy array and save
                    face_array = face_tensor.permute(1, 2, 0).cpu().numpy()
                    face_array = (face_array * 255).astype(np.uint8)
                    face_image = Image.fromarray(face_array)
                    
                    # Save face crop
                    frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
                    frame_path = output_dir / frame_filename
                    face_image.save(frame_path, quality=95)
                    extracted_count += 1
            except Exception as e:
                # Skip frame if face detection fails
                pass
        
        frame_count += 1
    
    cap.release()
    return extracted_count


def split_videos(video_list, train_split, val_split, test_split):
    """
    Split videos into train, validation, and test sets.
    Ensures no frame leakage by splitting at video level.
    
    Args:
        video_list: List of video paths
        train_split: Proportion for training
        val_split: Proportion for validation
        test_split: Proportion for testing
    
    Returns:
        Three lists: train_videos, val_videos, test_videos
    """
    random.shuffle(video_list)
    total = len(video_list)
    
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    
    train_videos = video_list[:train_end]
    val_videos = video_list[train_end:val_end]
    test_videos = video_list[val_end:]
    
    return train_videos, val_videos, test_videos


def process_videos(video_list, output_dir, label):
    """
    Process a list of videos and extract faces.
    
    Args:
        video_list: List of video paths
        output_dir: Directory to save extracted faces
        label: 'real' or 'fake'
    
    Returns:
        Total number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    
    for video_path in tqdm(video_list, desc=f"Processing {label} videos"):
        video_name = video_path.stem
        frames_extracted = extract_frames(video_path, output_dir, label, video_name)
        total_frames += frames_extracted
    
    return total_frames


def main():
    """Main preprocessing function."""
    print("=" * 60)
    print("DeepFake Detection - Preprocessing Pipeline")
    print("=" * 60)
    
    # Get all video files
    real_videos = list(config.REAL_DIR.glob("*.mp4"))
    fake_videos = list(config.FAKE_DIR.glob("*.mp4"))
    
    print(f"\nFound {len(real_videos)} real videos and {len(fake_videos)} fake videos")
    
    if len(real_videos) == 0 or len(fake_videos) == 0:
        raise ValueError("No videos found in DATASETS/REAL or DATASETS/FAKE directories")
    
    # Split videos (not frames) to avoid frame leakage
    print("\nSplitting videos into train/val/test sets...")
    real_train, real_val, real_test = split_videos(
        real_videos, config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT
    )
    fake_train, fake_val, fake_test = split_videos(
        fake_videos, config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT
    )
    
    print(f"Real videos - Train: {len(real_train)}, Val: {len(real_val)}, Test: {len(real_test)}")
    print(f"Fake videos - Train: {len(fake_train)}, Val: {len(fake_val)}, Test: {len(fake_test)}")
    
    # Process training set
    print("\n" + "=" * 60)
    print("Processing TRAINING set...")
    print("=" * 60)
    real_train_dir = config.TRAIN_DIR / "real"
    fake_train_dir = config.TRAIN_DIR / "fake"
    real_train_frames = process_videos(real_train, real_train_dir, "real")
    fake_train_frames = process_videos(fake_train, fake_train_dir, "fake")
    print(f"Training: {real_train_frames} real frames, {fake_train_frames} fake frames")
    
    # Process validation set
    print("\n" + "=" * 60)
    print("Processing VALIDATION set...")
    print("=" * 60)
    real_val_dir = config.VAL_DIR / "real"
    fake_val_dir = config.VAL_DIR / "fake"
    real_val_frames = process_videos(real_val, real_val_dir, "real")
    fake_val_frames = process_videos(fake_val, fake_val_dir, "fake")
    print(f"Validation: {real_val_frames} real frames, {fake_val_frames} fake frames")
    
    # Process test set
    print("\n" + "=" * 60)
    print("Processing TEST set...")
    print("=" * 60)
    real_test_dir = config.TEST_DIR / "real"
    fake_test_dir = config.TEST_DIR / "fake"
    real_test_frames = process_videos(real_test, real_test_dir, "real")
    fake_test_frames = process_videos(fake_test, fake_test_dir, "fake")
    print(f"Test: {real_test_frames} real frames, {fake_test_frames} fake frames")
    
    # Summary
    print("\n" + "=" * 60)
    print("Preprocessing Complete!")
    print("=" * 60)
    print(f"Total frames extracted:")
    print(f"  Training:   {real_train_frames + fake_train_frames:,} frames")
    print(f"  Validation: {real_val_frames + fake_val_frames:,} frames")
    print(f"  Test:       {real_test_frames + fake_test_frames:,} frames")
    print(f"  Grand Total: {real_train_frames + fake_train_frames + real_val_frames + fake_val_frames + real_test_frames + fake_test_frames:,} frames")
    print(f"\nData saved to: {config.DATA_DIR}")


if __name__ == "__main__":
    main()

