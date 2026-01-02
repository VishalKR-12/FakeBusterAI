"""
Dataset and data loading utilities for DeepFake Detection Pipeline
Includes data augmentation pipeline using albumentations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config


class DeepFakeDataset(Dataset):
    """
    Dataset class for loading face crops with labels.
    """
    
    def __init__(self, data_dir, transform=None, is_train=True):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing 'real' and 'fake' subdirectories
            transform: Albumentations transform pipeline
            is_train: Whether this is training data (affects augmentation)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_train = is_train
        
        # Load all image paths with labels
        self.samples = []
        
        # Real images (label 0)
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("*.jpg"):
                self.samples.append((img_path, 0))
        
        # Fake images (label 1)
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for img_path in fake_dir.glob("*.jpg"):
                self.samples.append((img_path, 1))
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        print(f"  Real: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"  Fake: {sum(1 for _, label in self.samples if label == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return image, torch.tensor(label, dtype=torch.long)


def get_train_transform():
    """
    Get training data augmentation pipeline.
    Includes: horizontal flip, Gaussian blur, JPEG compression, random crop, color jitter.
    """
    return A.Compose([
        A.HorizontalFlip(p=config.AUGMENTATION_PROB),
        A.GaussianBlur(
            blur_limit=(3, 7),
            sigma_limit=config.GAUSSIAN_BLUR_SIGMA_LIMIT,
            p=config.AUGMENTATION_PROB
        ),
        A.ImageCompression(
            quality_lower=config.JPEG_COMPRESSION_QUALITY_LOWER,
            quality_upper=config.JPEG_COMPRESSION_QUALITY_UPPER,
            p=config.AUGMENTATION_PROB
        ),
        A.RandomCrop(
            height=config.FACE_SIZE,
            width=config.FACE_SIZE,
            p=config.AUGMENTATION_PROB
        ),
        A.ColorJitter(
            brightness=config.COLOR_JITTER_BRIGHTNESS,
            contrast=config.COLOR_JITTER_CONTRAST,
            saturation=config.COLOR_JITTER_SATURATION,
            hue=config.COLOR_JITTER_HUE,
            p=config.AUGMENTATION_PROB
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transform():
    """
    Get validation/test data transform pipeline (no augmentation).
    """
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_data_loaders():
    """
    Create data loaders for train, validation, and test sets.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Training dataset with augmentation
    train_dataset = DeepFakeDataset(
        config.TRAIN_DIR,
        transform=get_train_transform(),
        is_train=True
    )
    
    # Validation dataset without augmentation
    val_dataset = DeepFakeDataset(
        config.VAL_DIR,
        transform=get_val_transform(),
        is_train=False
    )
    
    # Test dataset without augmentation
    test_dataset = DeepFakeDataset(
        config.TEST_DIR,
        transform=get_val_transform(),
        is_train=False
    )
    
    # Create data loaders
    # drop_last=True prevents BatchNorm errors with batch size 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True if config.DEVICE == "cuda" else False,
        drop_last=True  # Drop last incomplete batch to avoid BatchNorm issues
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.DEVICE == "cuda" else False,
        drop_last=True  # Drop last batch to avoid BatchNorm issues with batch size 1
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True if config.DEVICE == "cuda" else False,
        drop_last=True  # Drop last batch to avoid BatchNorm issues with batch size 1
    )
    
    return train_loader, val_loader, test_loader

