"""
Evaluation script for DeepFake Detection Pipeline
Evaluates model on test set with frame-level and video-level metrics.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import config
from model import create_model
from dataset import DeepFakeDataset, get_val_transform

# Set device
device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    
    return model


def extract_video_name(frame_path):
    """Extract video name from frame filename."""
    # Frame format: {video_name}_frame_{frame_number}.jpg
    frame_name = Path(frame_path).stem
    video_name = '_'.join(frame_name.split('_frame_')[:-1])
    return video_name


def evaluate_frame_level(model, test_dataset, device, use_amp=False):
    """Evaluate model at frame level."""
    print("\n" + "=" * 60)
    print("Frame-Level Evaluation")
    print("=" * 60)
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_frame_paths = []
    
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluating frames"):
            image, label = test_dataset[idx]
            image = image.unsqueeze(0).to(device)
            
            if use_amp:
                with autocast():
                    outputs = model(image)
            else:
                outputs = model(image)
            
            probs = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
            
            all_preds.append(pred.cpu().item())
            all_probs.append(probs[0, 1].cpu().item())  # Probability of fake class
            all_labels.append(label.item())
            all_frame_paths.append(test_dataset.samples[idx][0])
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\nFrame-Level Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'frame_paths': [str(p) for p in all_frame_paths]
    }


def evaluate_video_level(frame_results, aggregation_method='mean'):
    """Aggregate frame-level predictions to video-level."""
    print("\n" + "=" * 60)
    print(f"Video-Level Evaluation (Aggregation: {aggregation_method})")
    print("=" * 60)
    
    # Group frames by video
    video_data = defaultdict(lambda: {'probs': [], 'preds': [], 'labels': []})
    
    for i, frame_path in enumerate(frame_results['frame_paths']):
        video_name = extract_video_name(frame_path)
        video_data[video_name]['probs'].append(frame_results['probabilities'][i])
        video_data[video_name]['preds'].append(frame_results['predictions'][i])
        video_data[video_name]['labels'].append(frame_results['labels'][i])
    
    # Aggregate predictions
    video_preds = []
    video_probs = []
    video_labels = []
    video_names = []
    
    for video_name, data in video_data.items():
        if aggregation_method == 'mean':
            # Mean probability
            mean_prob = np.mean(data['probs'])
            video_prob = mean_prob
            video_pred = 1 if mean_prob >= config.CONFIDENCE_THRESHOLD else 0
        elif aggregation_method == 'majority':
            # Majority vote
            video_pred = int(np.bincount(data['preds']).argmax())
            video_prob = np.mean(data['probs'])  # Still use mean prob for confidence
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        video_preds.append(video_pred)
        video_probs.append(video_prob)
        video_labels.append(data['labels'][0])  # All frames from same video have same label
        video_names.append(video_name)
    
    # Calculate metrics
    accuracy = accuracy_score(video_labels, video_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        video_labels, video_preds, average='binary', zero_division=0
    )
    try:
        roc_auc = roc_auc_score(video_labels, video_probs)
    except:
        roc_auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(video_labels, video_preds)
    
    print(f"\nVideo-Level Metrics:")
    print(f"  Number of videos: {len(video_names)}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'predictions': video_preds,
        'probabilities': video_probs,
        'labels': video_labels,
        'video_names': video_names
    }


def plot_confusion_matrix(cm, labels, title, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_comparison(frame_results, video_results, save_path):
    """Plot comparison between frame-level and video-level metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    frame_values = [frame_results[m] for m in metrics]
    video_values = [video_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, frame_values, width, label='Frame-Level', alpha=0.8)
    bars2 = ax.bar(x + width/2, video_values, width, label='Video-Level', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Frame-Level vs Video-Level Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {save_path}")


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("DeepFake Detection - Evaluation Pipeline")
    print("=" * 60)
    
    # Load model
    model_path = config.MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    model = load_model(model_path)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = DeepFakeDataset(
        config.TEST_DIR,
        transform=get_val_transform(),
        is_train=False
    )
    
    # Check for mixed precision
    use_amp = config.MIXED_PRECISION and device.type == 'cuda'
    
    # Frame-level evaluation
    frame_results = evaluate_frame_level(model, test_dataset, device, use_amp)
    
    # Video-level evaluation
    video_results_mean = evaluate_video_level(frame_results, aggregation_method='mean')
    video_results_majority = evaluate_video_level(frame_results, aggregation_method='majority')
    
    # Save results
    results = {
        'frame_level': frame_results,
        'video_level_mean': video_results_mean,
        'video_level_majority': video_results_majority
    }
    
    results_path = config.RESULTS_DIR / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Plot confusion matrices
    cm_frame = np.array(frame_results['confusion_matrix'])
    plot_confusion_matrix(
        cm_frame,
        ['Real', 'Fake'],
        'Frame-Level Confusion Matrix',
        config.RESULTS_DIR / "confusion_matrix_frame.png"
    )
    
    cm_video = np.array(video_results_mean['confusion_matrix'])
    plot_confusion_matrix(
        cm_video,
        ['Real', 'Fake'],
        'Video-Level Confusion Matrix (Mean Aggregation)',
        config.RESULTS_DIR / "confusion_matrix_video.png"
    )
    
    # Plot comparison
    plot_comparison(
        frame_results,
        video_results_mean,
        config.RESULTS_DIR / "frame_vs_video_comparison.png"
    )
    
    # Print classification report
    print("\n" + "=" * 60)
    print("Frame-Level Classification Report")
    print("=" * 60)
    print(classification_report(
        frame_results['labels'],
        frame_results['predictions'],
        target_names=['Real', 'Fake']
    ))
    
    print("\n" + "=" * 60)
    print("Video-Level Classification Report (Mean Aggregation)")
    print("=" * 60)
    print(classification_report(
        video_results_mean['labels'],
        video_results_mean['predictions'],
        target_names=['Real', 'Fake']
    ))
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

