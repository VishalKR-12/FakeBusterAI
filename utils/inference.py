"""
Model inference utilities for deepfake detection.
"""

import torch
from torch.cuda.amp import autocast
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

import config
from model import create_model


# Image transform (same as validation)
transform = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])


def load_model(model_path: str, device=None):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (.pth file)
        device: Device to load model on (default: auto-detect)
    
    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # Suppress FutureWarning about weights_only (we trust our own model files)
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
    
    return model, device


def preprocess_face(face_image: Image.Image) -> torch.Tensor:
    """
    Preprocess face image for model inference.
    
    Args:
        face_image: PIL Image of face (224x224)
    
    Returns:
        Preprocessed tensor ready for model
    """
    # Convert to numpy array
    face_array = np.array(face_image)
    
    # Apply transforms
    transformed = transform(image=face_array)
    tensor = transformed["image"]
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor


def predict_single_face(model, face_tensor, device, use_amp=False):
    """
    Predict on a single face image.
    
    Args:
        model: Trained model
        face_tensor: Preprocessed face tensor
        device: Device to run inference on
        use_amp: Use mixed precision
    
    Returns:
        Dictionary with predictions:
        - real_prob: Probability of being real
        - fake_prob: Probability of being fake
        - prediction: 0 (real) or 1 (fake)
        - confidence: Confidence score
    """
    face_tensor = face_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        if use_amp:
            with autocast():
                outputs = model(face_tensor)
        else:
            outputs = model(face_tensor)
        
        probs = torch.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, 1)
    
    real_prob = probs[0, 0].cpu().item()
    fake_prob = probs[0, 1].cpu().item()
    prediction = pred[0].cpu().item()
    confidence = fake_prob if prediction == 1 else real_prob
    
    return {
        'real_prob': real_prob,
        'fake_prob': fake_prob,
        'prediction': prediction,
        'confidence': confidence
    }


def predict_batch(model, face_tensors, device, use_amp=False):
    """
    Predict on a batch of face images.
    
    Args:
        model: Trained model
        face_tensors: Batch of preprocessed face tensors
        device: Device to run inference on
        use_amp: Use mixed precision
    
    Returns:
        List of prediction dictionaries
    """
    face_tensors = face_tensors.to(device)
    
    model.eval()
    with torch.no_grad():
        if use_amp:
            with autocast():
                outputs = model(face_tensors)
        else:
            outputs = model(face_tensors)
        
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    results = []
    for i in range(len(face_tensors)):
        real_prob = probs[i, 0].cpu().item()
        fake_prob = probs[i, 1].cpu().item()
        prediction = preds[i].cpu().item()
        confidence = fake_prob if prediction == 1 else real_prob
        
        results.append({
            'real_prob': real_prob,
            'fake_prob': fake_prob,
            'prediction': prediction,
            'confidence': confidence
        })
    
    return results


def aggregate_video_predictions(frame_results, method='mean'):
    """
    Aggregate frame-level predictions to video-level verdict.
    
    Args:
        frame_results: List of frame result dictionaries (each with 'prediction' key)
        method: Aggregation method ('mean' or 'majority')
    
    Returns:
        Dictionary with video-level prediction
    """
    if len(frame_results) == 0:
        return {
            'verdict': 'REAL',
            'confidence': 0.5,
            'real_prob': 0.5,
            'fake_prob': 0.5,
            'num_frames': 0
        }
    
    # Extract predictions from frame results
    frame_predictions = [r['prediction'] if 'prediction' in r else r for r in frame_results]
    
    fake_probs = [p['fake_prob'] for p in frame_predictions]
    real_probs = [p['real_prob'] for p in frame_predictions]
    
    if method == 'mean':
        avg_fake_prob = np.mean(fake_probs)
        avg_real_prob = np.mean(real_probs)
        verdict = 'FAKE' if avg_fake_prob >= config.CONFIDENCE_THRESHOLD else 'REAL'
        confidence = avg_fake_prob if verdict == 'FAKE' else avg_real_prob
    elif method == 'majority':
        predictions = [p['prediction'] for p in frame_predictions]
        majority = int(np.bincount(predictions).argmax())
        avg_fake_prob = np.mean(fake_probs)
        avg_real_prob = np.mean(real_probs)
        verdict = 'FAKE' if majority == 1 else 'REAL'
        confidence = avg_fake_prob if verdict == 'FAKE' else avg_real_prob
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return {
        'verdict': verdict,
        'confidence': confidence,
        'real_prob': avg_real_prob,
        'fake_prob': avg_fake_prob,
        'num_frames': len(frame_predictions)
    }

