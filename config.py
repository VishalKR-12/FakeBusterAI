"""
Configuration file for DeepFake Detection Pipeline
All hyperparameters and paths are centralized here for easy modification.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "DATASETS"
REAL_DIR = DATASET_DIR / "REAL"
FAKE_DIR = DATASET_DIR / "FAKE"

# Output directories
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = OUTPUT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories
for dir_path in [OUTPUT_DIR, DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Preprocessing parameters
FPS = 5  # Frames per second for extraction
FACE_SIZE = 224  # Target face crop size
FACE_DETECTION_METHOD = "mtcnn"  # Options: "mtcnn" or "retinaface"
MIN_FACE_SIZE = 50  # Minimum face size to accept
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Data augmentation parameters
AUGMENTATION_PROB = 0.5
GAUSSIAN_BLUR_SIGMA_LIMIT = (0.1, 2.0)
JPEG_COMPRESSION_QUALITY_LOWER = 50
JPEG_COMPRESSION_QUALITY_UPPER = 100
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.1

# Model parameters
MODEL_NAME = "xception"  # Options: "xception", "efficientnet_b0", "efficientnet_b3", "resnet50"
PRETRAINED = True
NUM_CLASSES = 2
DROPOUT_RATE = 0.5

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10  # Reduced from 30 for faster training
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MIXED_PRECISION = True  # Use automatic mixed precision if GPU available

# Optimizer and scheduler
OPTIMIZER = "adamw"  # Options: "adamw", "adam", "sgd"
SCHEDULER = "reduce_lr_on_plateau"  # Options: "reduce_lr_on_plateau", "cosine", "step"
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6

# Early stopping
EARLY_STOPPING_PATIENCE = 7
EARLY_STOPPING_MIN_DELTA = 0.001

# Reproducibility
RANDOM_SEED = 42

# Device (scripts should check torch.cuda.is_available() before using)
# This is just a default - actual device detection happens in each script
DEVICE = "cuda"  # Will fallback to "cpu" if CUDA not available

# Inference parameters
INFERENCE_BATCH_SIZE = 16
AGGREGATION_METHOD = "mean"  # Options: "mean" (mean probability) or "majority" (majority vote)
CONFIDENCE_THRESHOLD = 0.5

# Evaluation metrics
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Checkpoint saving
SAVE_BEST_MODEL = True
CHECKPOINT_FREQUENCY = 5  # Save checkpoint every N epochs

