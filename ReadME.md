# DeepFake Detection Pipeline

A production-ready, research-grade deepfake detection system that processes video clips, extracts facial regions, trains a binary classifier, and aggregates frame-level predictions to video-level classifications. This pipeline is designed for reproducibility, modularity, and immediate deployment.

## Features

- **Robust Preprocessing**: Frame extraction at 5 FPS with MTCNN face detection and 224×224 face cropping
- **Video-Level Data Splitting**: Ensures no frame leakage by splitting at video level (70/15/15 train/val/test)
- **Transfer Learning**: Supports Xception, EfficientNet-B0/B3, and ResNet-50 with ImageNet pretrained weights
- **Advanced Training**: Mixed precision training, early stopping, learning rate scheduling, and comprehensive data augmentation
- **Frame-to-Video Aggregation**: Mean probability or majority vote for video-level predictions
- **Comprehensive Evaluation**: Frame-level and video-level metrics with detailed visualizations
- **Production-Ready Inference**: Process new videos with confidence scores and detailed output
- **Interactive Web Application**: Streamlit-based UI for real-time deepfake detection with explainable AI visualization

## Dataset Structure

Place your videos in the following structure:

```
DATASETS/
├── REAL/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── FAKE/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

**Requirements:**
- 300 real videos and 300 fake videos (or any balanced dataset)
- Videos should be 5-10 seconds long with visible faces
- Supported formats: `.mp4`, `.avi`, `.mov`

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- OpenCV >= 4.8.0
- facenet-pytorch >= 2.5.3 (for MTCNN face detection)
- albumentations >= 1.3.1 (for data augmentation)
- scikit-learn >= 1.3.0 (for metrics)
- timm >= 0.9.0 (for EfficientNet and Xception models)
- streamlit >= 1.28.0 (for interactive web application)

## Quick Start

### Option A: Command-Line Pipeline

Follow steps 1-4 below for the complete training and evaluation pipeline.

### Option B: Interactive Web Application

If you already have a trained model, jump directly to step 5 to use the Streamlit web app.

---

### 1. Preprocessing

Extract frames, detect faces, and split data:

```bash
python preprocess.py
```

This will:
- Extract frames from all videos at 5 FPS
- Detect faces using MTCNN
- Crop faces to 224×224 pixels
- Split videos into train/val/test sets (70/15/15) ensuring no frame leakage
- Save processed data to `output/data/`

**Output structure:**
```
output/data/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### 2. Training

Train the model:

```bash
python train.py
```

**Training features:**
- Transfer learning with ImageNet pretrained weights
- AdamW optimizer (learning rate: 3e-4)
- ReduceLROnPlateau scheduler
- Mixed precision training (if GPU available)
- Early stopping to prevent overfitting
- Automatic checkpoint saving

**Training outputs:**
- Best model: `output/models/best_model.pth`
- Training history: `output/logs/training_history.json`
- Training curves: `output/results/training_curves.png`

### 3. Evaluation

Evaluate the trained model on the test set:

```bash
python evaluate.py
```

**Evaluation outputs:**
- Frame-level and video-level metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrices for both levels
- Comparison plots
- Detailed classification reports
- Results saved to `output/results/evaluation_results.json`

### 4. Inference

Run inference on new videos:

```bash
# Single video
python inference.py path/to/video.mp4

# Directory of videos
python inference.py path/to/videos/

# Custom model and output path
python inference.py path/to/video.mp4 --model output/models/best_model.pth --output results.txt
```

**Inference output:**
- Video name
- Prediction: REAL or FAKE
- Confidence score (0-1)
- Number of frames processed

### 5. Interactive Web Application (Streamlit)

Launch the interactive web application for real-time deepfake detection:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

#### 🎯 Streamlit App Features

**1. User Interface**
- Clean, modern UI with color-coded verdicts
- Upload widget supporting images (.jpg, .png) and videos (.mp4, .avi, .mov)
- Real-time progress indicators during processing
- Expandable sections for detailed analysis

**2. Image Processing Flow**
- Automatic face detection using MTCNN
- Face cropping and resizing to 224×224
- Model inference with confidence scores
- Visual display of:
  - Original image
  - Detected and cropped face
  - Prediction label (REAL/FAKE) with confidence percentage
  - Detailed probability breakdown

**3. Video Processing Flow**
- Frame extraction at configurable FPS (1-10 FPS)
- Face detection on each frame
- Frame-by-frame analysis display showing:
  - Original frame
  - Cropped face
  - Real and Fake probabilities
  - Frame-level prediction
- Video-level aggregation with final verdict

**4. Frame-Level Explainability**
For each processed frame, the app displays:
- Frame number
- Fake Probability: X.XX
- Real Probability: X.XX
- Prediction: REAL or FAKE
- Confidence score

**5. Video-Level Aggregation**
- Computes final verdict using mean probability across all frames
- Threshold-based classification (≥ 0.5 → FAKE, < 0.5 → REAL)
- Displays:
  - Total frames analyzed
  - Average confidence score
  - Final verdict badge (color-coded: 🔴 Red for FAKE, 🟢 Green for REAL)

**6. Advanced Features**
- **Filter Options**: Show only FAKE frames with minimum confidence threshold
- **Frame Sampling Rate**: Adjustable FPS for video processing (1-10 FPS)
- **Download Reports**: Generate and download text reports of analysis
- **Model Path Configuration**: Specify custom model checkpoint path
- **Real-time Processing**: Progress bars and status updates during analysis

**7. Sidebar Configuration**
- Model path selection
- Frame sampling rate slider (1-10 FPS)
- Filter options (show only fake frames)
- Minimum confidence threshold

#### 📋 Step-by-Step Usage Guide

**Step 1: Launch the Application**
```bash
streamlit run app.py
```

**Step 2: Configure Settings (Optional)**
- In the sidebar, verify the model path points to your trained model
- Adjust frame sampling rate if processing videos (default: 5 FPS)
- Set filter options if you want to focus on specific frames

**Step 3: Upload Media**
- Click "Browse files" or drag and drop an image or video
- Supported formats:
  - Images: `.jpg`, `.jpeg`, `.png`
  - Videos: `.mp4`, `.avi`, `.mov`

**Step 4: View Results**
- **For Images**: 
  - See original image and detected face side-by-side
  - View final verdict with confidence percentage
  - Expand "Detailed Probabilities" for breakdown
  
- **For Videos**:
  - See video-level summary with final verdict
  - View frame-by-frame analysis table
  - Browse frame visualizations in grid layout
  - Each frame shows original, cropped face, and predictions

**Step 5: Download Report (Optional)**
- Click "Download Report" button to save analysis results as text file
- Report includes all predictions and confidence scores

#### 🎨 Visual Features

- **Color-Coded Verdicts**: 
  - 🔴 Red background for FAKE predictions
  - 🟢 Green background for REAL predictions
  
- **Progress Indicators**: Real-time progress bars during processing
- **Frame Grid Display**: Visual grid showing all processed frames
- **Interactive Tables**: Sortable and filterable frame analysis data
- **Responsive Design**: Works on different screen sizes

#### ⚙️ Technical Details

**Model Loading:**
- Models are cached using `@st.cache_resource` for efficient reloading
- Automatic GPU detection and fallback to CPU
- Supports all model architectures (Xception, EfficientNet, ResNet-50)

**Processing Pipeline:**
1. File upload and type detection
2. Face detection using MTCNN
3. Image preprocessing (normalization, resizing)
4. Model inference with mixed precision (if GPU available)
5. Result aggregation and visualization

**Performance:**
- Efficient batch processing for video frames
- GPU acceleration when available
- Optimized face detection pipeline
- Memory-efficient frame extraction

#### 🐛 Troubleshooting Streamlit App

**App won't start:**
- Ensure Streamlit is installed: `pip install streamlit`
- Check that model file exists at specified path
- Verify all dependencies are installed

**No faces detected:**
- Ensure uploaded media contains visible faces
- Try different images/videos
- Check video quality and resolution

**Slow processing:**
- Reduce frame sampling rate for videos
- Use GPU if available (automatic detection)
- Process shorter videos for faster results

**Model loading errors:**
- Verify model path in sidebar
- Ensure model was trained with compatible architecture
- Check model file integrity

## Configuration

All hyperparameters and paths are centralized in `config.py`. Key settings:

### Model Configuration
- `MODEL_NAME`: Choose from `"xception"`, `"efficientnet_b0"`, `"efficientnet_b3"`, `"resnet50"`
- `PRETRAINED`: Use ImageNet pretrained weights (default: `True`)
- `DROPOUT_RATE`: Dropout rate in classifier head (default: `0.5`)

### Training Configuration
- `BATCH_SIZE`: Batch size (default: `32`)
- `NUM_EPOCHS`: Maximum number of epochs (default: `30`)
- `LEARNING_RATE`: Initial learning rate (default: `3e-4`)
- `MIXED_PRECISION`: Use automatic mixed precision (default: `True`)

### Preprocessing Configuration
- `FPS`: Frames per second for extraction (default: `5`)
- `FACE_SIZE`: Target face crop size (default: `224`)
- `FACE_DETECTION_METHOD`: Face detector (default: `"mtcnn"`)

### Inference Configuration
- `AGGREGATION_METHOD`: `"mean"` (mean probability) or `"majority"` (majority vote)
- `CONFIDENCE_THRESHOLD`: Threshold for binary classification (default: `0.5`)

## Data Augmentation

The training pipeline includes comprehensive data augmentation:

- **Horizontal Flip**: Random horizontal flipping
- **Gaussian Blur**: Random Gaussian blurring
- **JPEG Compression**: Random JPEG compression artifacts
- **Random Crop**: Random cropping
- **Color Jitter**: Random brightness, contrast, saturation, and hue adjustments

All augmentation parameters are configurable in `config.py`.

## Model Architectures

### Xception (Recommended)
- Strong performance on image classification tasks
- Efficient depthwise separable convolutions
- 2048 feature dimensions

### EfficientNet-B0/B3
- State-of-the-art efficiency/accuracy trade-off
- B0: 1280 features, faster training
- B3: 1536 features, better accuracy

### ResNet-50
- Proven architecture with strong transfer learning capabilities
- 2048 feature dimensions
- Fast and reliable

## Evaluation Metrics

The evaluation pipeline computes:

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification breakdown

Metrics are computed at both:
1. **Frame-level**: Individual frame predictions
2. **Video-level**: Aggregated video predictions

## Project Structure

```
DeepFakeDetection/
├── DATASETS/              # Input videos
│   ├── REAL/
│   └── FAKE/
├── output/                # Generated outputs
│   ├── data/              # Preprocessed frames
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── models/            # Saved model checkpoints
│   │   ├── best_model.pth
│   │   └── checkpoint_epoch_*.pth
│   ├── logs/              # Training logs
│   │   └── training_history.json
│   └── results/           # Evaluation results
│       ├── training_curves.png
│       ├── confusion_matrix_frame.png
│       ├── confusion_matrix_video.png
│       ├── frame_vs_video_comparison.png
│       ├── evaluation_results.json
│       └── inference_results.txt
├── utils/                 # Utility modules
│   ├── __init__.py
│   ├── face_utils.py      # Face detection utilities
│   ├── video_utils.py     # Video processing utilities
│   └── inference.py      # Model inference utilities
├── app.py                 # Streamlit web application
├── config.py              # Configuration file
├── dataset.py             # Dataset and data loading
├── model.py               # Model architecture
├── preprocess.py          # Data preprocessing script
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── inference.py           # Command-line inference script
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .gitignore            # Git ignore rules
```

## Output Structure

```
output/
├── data/              # Preprocessed frames
│   ├── train/
│   ├── val/
│   └── test/
├── models/             # Saved model checkpoints
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
├── logs/               # Training logs
│   └── training_history.json
└── results/            # Evaluation and inference results
    ├── training_curves.png
    ├── confusion_matrix_frame.png
    ├── confusion_matrix_video.png
    ├── frame_vs_video_comparison.png
    ├── evaluation_results.json
    └── inference_results.txt
```

## Reproducibility

The pipeline ensures reproducibility through:

- Fixed random seeds (configurable in `config.py`)
- Deterministic CUDA operations (when available)
- Video-level data splitting (prevents frame leakage)
- Comprehensive logging of all hyperparameters

## Performance Tips

1. **GPU Acceleration**: The pipeline automatically uses GPU if available. Set `CUDA_VISIBLE_DEVICES` to specify GPU.

2. **Batch Size**: Adjust `BATCH_SIZE` in `config.py` based on available GPU memory. Larger batches may improve training stability.

3. **Mixed Precision**: Enabled by default for faster training and lower memory usage on modern GPUs.

4. **Data Loading**: Uses 4 workers by default. Adjust in `dataset.py` if needed.

5. **Face Detection**: MTCNN is robust but can be slow. Consider using RetinaFace for faster processing (requires implementation).

## Troubleshooting

### No faces detected in videos
- Ensure videos contain visible faces
- Check `MIN_FACE_SIZE` in `config.py` (may be too large)
- Verify video quality and resolution

### Out of memory errors
- Reduce `BATCH_SIZE` in `config.py`
- Reduce number of data loader workers
- Disable mixed precision training

### Poor model performance
- Increase training epochs
- Adjust learning rate
- Try different model architecture
- Check data augmentation settings
- Ensure balanced dataset

### Slow preprocessing
- Face detection is computationally intensive
- Consider processing videos in batches
- Use GPU for face detection (MTCNN supports GPU)

## Suggested Improvements

### 1. Temporal Models
- **3D CNNs**: Process video sequences directly to capture temporal patterns
- **LSTM/GRU**: Add recurrent layers to model temporal dependencies
- **Two-stream networks**: Combine spatial and temporal streams

### 2. Transformer-Based Approaches
- **Vision Transformers (ViT)**: Apply transformer architecture to face patches
- **Video Transformers**: Model temporal relationships with attention mechanisms
- **Hybrid CNN-Transformer**: Combine CNN features with transformer attention

### 3. Advanced Techniques
- **Ensemble Methods**: Combine predictions from multiple models
- **Self-Supervised Learning**: Pre-train on unlabeled video data
- **Adversarial Training**: Improve robustness to adversarial examples
- **Attention Mechanisms**: Focus on discriminative facial regions
- **Multi-scale Features**: Process faces at multiple resolutions

### 4. Data Improvements
- **Synthetic Data Generation**: Use GANs to augment training data
- **Hard Negative Mining**: Focus training on difficult examples
- **Domain Adaptation**: Adapt to different video sources and qualities

### 5. Post-Processing
- **Temporal Smoothing**: Apply smoothing filters to frame-level predictions
- **Confidence Calibration**: Improve confidence score reliability
- **Uncertainty Quantification**: Provide uncertainty estimates

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{deepfake_detection_pipeline,
  title = {DeepFake Detection Pipeline},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/deepfake-detection}
}
```

## License

This project is provided as-is for research and educational purposes.

## Contact

For questions, issues, or contributions, please open an issue on the repository.

---

**Note**: This pipeline is designed for research and production use. Ensure compliance with local regulations and ethical guidelines when deploying deepfake detection systems.

