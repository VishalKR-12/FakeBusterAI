"""
Inference script for DeepFake Detection Pipeline
Processes new videos and outputs video-level predictions with confidence scores.
"""

import torch
from torch.cuda.amp import autocast
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from model import create_model

# Set device
device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
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
    raise ValueError(f"Face detection method '{config.FACE_DETECTION_METHOD}' not implemented.")

# Image transform (same as validation)
transform = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])


def load_model(model_path):
    """Load trained model from checkpoint."""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model with same architecture
    model = create_model(
        model_name=checkpoint['config']['model_name'],
        pretrained=False,
        num_classes=checkpoint['config']['num_classes'],
        dropout_rate=checkpoint['config']['dropout_rate']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def extract_frames_from_video(video_path):
    """
    Extract frames from video at specified FPS and detect faces.
    
    Returns:
        List of face crops (PIL Images)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / config.FPS))
    frame_count = 0
    face_crops = []
    
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
                    # Convert tensor to numpy array
                    face_array = face_tensor.permute(1, 2, 0).cpu().numpy()
                    face_array = (face_array * 255).astype(np.uint8)
                    face_image = Image.fromarray(face_array)
                    face_crops.append(face_image)
            except Exception as e:
                # Skip frame if face detection fails
                pass
        
        frame_count += 1
    
    cap.release()
    return face_crops


def predict_frames(model, face_crops, device, use_amp=False):
    """
    Predict on a batch of face crops.
    
    Returns:
        List of probabilities (probability of fake class)
    """
    if len(face_crops) == 0:
        return []
    
    # Preprocess images
    images = []
    for face_crop in face_crops:
        face_array = np.array(face_crop)
        transformed = transform(image=face_array)
        images.append(transformed["image"])
    
    # Batch images
    images_tensor = torch.stack(images).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        if use_amp:
            with autocast():
                outputs = model(images_tensor)
        else:
            outputs = model(images_tensor)
        
        probs = torch.softmax(outputs, dim=1)
        fake_probs = probs[:, 1].cpu().numpy()  # Probability of fake class
    
    return fake_probs.tolist()


def aggregate_predictions(frame_probs, aggregation_method='mean'):
    """
    Aggregate frame-level predictions to video-level.
    
    Args:
        frame_probs: List of frame-level probabilities
        aggregation_method: 'mean' or 'majority'
    
    Returns:
        video_prediction: 0 (Real) or 1 (Fake)
        video_confidence: Confidence score (0-1)
    """
    if len(frame_probs) == 0:
        return 0, 0.5  # Default to Real with neutral confidence
    
    if aggregation_method == 'mean':
        mean_prob = np.mean(frame_probs)
        video_pred = 1 if mean_prob >= config.CONFIDENCE_THRESHOLD else 0
        video_confidence = mean_prob if video_pred == 1 else (1 - mean_prob)
    elif aggregation_method == 'majority':
        frame_preds = [1 if p >= config.CONFIDENCE_THRESHOLD else 0 for p in frame_probs]
        video_pred = int(np.bincount(frame_preds).argmax())
        mean_prob = np.mean(frame_probs)
        video_confidence = mean_prob if video_pred == 1 else (1 - mean_prob)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    return video_pred, video_confidence


def process_video(video_path, model, device, use_amp=False):
    """
    Process a single video and return prediction.
    
    Returns:
        prediction: 'REAL' or 'FAKE'
        confidence: Confidence score (0-1)
        num_frames: Number of frames processed
    """
    # Extract frames and detect faces
    face_crops = extract_frames_from_video(video_path)
    
    if len(face_crops) == 0:
        print(f"Warning: No faces detected in {video_path}")
        return "REAL", 0.5, 0
    
    # Process in batches
    all_probs = []
    batch_size = config.INFERENCE_BATCH_SIZE
    
    for i in range(0, len(face_crops), batch_size):
        batch = face_crops[i:i + batch_size]
        batch_probs = predict_frames(model, batch, device, use_amp)
        all_probs.extend(batch_probs)
    
    # Aggregate to video-level
    video_pred, video_confidence = aggregate_predictions(
        all_probs,
        aggregation_method=config.AGGREGATION_METHOD
    )
    
    prediction = "FAKE" if video_pred == 1 else "REAL"
    
    return prediction, video_confidence, len(face_crops)


def main():
    """Main inference function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepFake Detection Inference")
    parser.add_argument(
        "input",
        type=str,
        help="Path to video file or directory containing videos"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(config.MODEL_DIR / "best_model.pth"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(config.RESULTS_DIR / "inference_results.txt"),
        help="Path to output file"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DeepFake Detection - Inference Pipeline")
    print("=" * 60)
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = load_model(model_path)
    
    # Check for mixed precision
    use_amp = config.MIXED_PRECISION and device.type == 'cuda'
    
    # Get input videos
    input_path = Path(args.input)
    if input_path.is_file():
        video_paths = [input_path]
    elif input_path.is_dir():
        video_paths = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi")) + list(input_path.glob("*.mov"))
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    if len(video_paths) == 0:
        raise ValueError(f"No videos found at {input_path}")
    
    print(f"\nProcessing {len(video_paths)} video(s)...")
    print("=" * 60)
    
    # Process videos
    results = []
    for video_path in tqdm(video_paths, desc="Processing videos"):
        try:
            prediction, confidence, num_frames = process_video(
                video_path, model, device, use_amp
            )
            results.append({
                'video': str(video_path),
                'prediction': prediction,
                'confidence': confidence,
                'num_frames': num_frames
            })
            
            print(f"\n{video_path.name}:")
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Frames processed: {num_frames}")
        except Exception as e:
            print(f"\nError processing {video_path}: {e}")
            results.append({
                'video': str(video_path),
                'prediction': 'ERROR',
                'confidence': 0.0,
                'num_frames': 0,
                'error': str(e)
            })
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("DeepFake Detection - Inference Results\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            f.write(f"Video: {result['video']}\n")
            f.write(f"  Prediction: {result['prediction']}\n")
            f.write(f"  Confidence: {result['confidence']:.4f}\n")
            f.write(f"  Frames processed: {result['num_frames']}\n")
            if 'error' in result:
                f.write(f"  Error: {result['error']}\n")
            f.write("\n")
    
    print(f"\nResults saved to {output_path}")
    print("=" * 60)
    print("Inference Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

