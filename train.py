"""
Training script for DeepFake Detection Pipeline
Includes mixed precision training, early stopping, checkpoint saving, and learning rate scheduling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import config
from model import create_model, count_parameters
from dataset import get_data_loaders

# Set random seeds for reproducibility
torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device
device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, use_amp=False):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, use_amp=False):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)
            
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0
    
    return epoch_loss, epoch_acc, precision, recall, f1, roc_auc


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_acc': best_val_acc,
        'config': {
            'model_name': config.MODEL_NAME,
            'num_classes': config.NUM_CLASSES,
            'dropout_rate': config.DROPOUT_RATE,
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def plot_training_curves(history, save_path):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curve
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curve
    axes[0, 1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 score curve
    axes[1, 0].plot(history['val_f1'], label='Val F1', marker='s', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # ROC-AUC curve
    axes[1, 1].plot(history['val_roc_auc'], label='Val ROC-AUC', marker='s', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].set_title('Validation ROC-AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def main():
    """Main training function."""
    print("=" * 60)
    print("DeepFake Detection - Training Pipeline")
    print("=" * 60)
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = get_data_loaders()
    
    # Create model
    print(f"\nCreating model: {config.MODEL_NAME}")
    model = create_model()
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if config.OPTIMIZER.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            momentum=0.9
        )
    
    # Scheduler
    if config.SCHEDULER.lower() == "reduce_lr_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.SCHEDULER_MIN_LR,
            verbose=True
        )
    else:
        scheduler = None
    
    # Mixed precision training
    use_amp = config.MIXED_PRECISION and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode='max'
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_roc_auc': []
    }
    
    best_val_acc = 0.0
    best_model_path = config.MODEL_DIR / "best_model.pth"
    
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    print("=" * 60)
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, val_roc_auc = validate(
            model, val_loader, criterion, device, use_amp
        )
        
        # Update learning rate
        if scheduler:
            if config.SCHEDULER.lower() == "reduce_lr_on_plateau":
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['val_roc_auc'].append(val_roc_auc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        print(f"Val F1: {val_f1:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, best_model_path)
            print(f"New best model saved! Val Acc: {best_val_acc:.4f}")
        
        # Save periodic checkpoint
        if epoch % config.CHECKPOINT_FREQUENCY == 0:
            checkpoint_path = config.MODEL_DIR / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, checkpoint_path)
        
        # Early stopping
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Save training history
    history_path = config.LOG_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    # Plot training curves
    curves_path = config.RESULTS_DIR / "training_curves.png"
    plot_training_curves(history, curves_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()

