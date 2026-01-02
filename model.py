"""
Model architecture definitions for DeepFake Detection Pipeline
Supports Xception, EfficientNet-B0/B3, and ResNet-50 with transfer learning.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm  # For EfficientNet and Xception
import config


class DeepFakeDetector(nn.Module):
    """
    DeepFake detection model with transfer learning.
    Supports multiple backbone architectures.
    """
    
    def __init__(self, model_name=config.MODEL_NAME, pretrained=config.PRETRAINED, 
                 num_classes=config.NUM_CLASSES, dropout_rate=config.DROPOUT_RATE):
        """
        Initialize model.
        
        Args:
            model_name: Name of backbone architecture
            pretrained: Whether to use pretrained weights
            num_classes: Number of output classes (2 for binary classification)
            dropout_rate: Dropout rate in classifier head
        """
        super(DeepFakeDetector, self).__init__()
        
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        
        # Load backbone architecture
        if self.model_name == "xception":
            self.backbone = self._create_xception(pretrained)
            num_features = 2048
        elif self.model_name == "efficientnet_b0":
            self.backbone = self._create_efficientnet_b0(pretrained)
            num_features = 1280
        elif self.model_name == "efficientnet_b3":
            self.backbone = self._create_efficientnet_b3(pretrained)
            num_features = 1536
        elif self.model_name == "resnet50":
            self.backbone = self._create_resnet50(pretrained)
            num_features = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}. Choose from: xception, efficientnet_b0, efficientnet_b3, resnet50")
        
        # Classifier head with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def _create_xception(self, pretrained):
        """Create Xception backbone."""
        try:
            # Use legacy_xception to avoid deprecation warning
            model = timm.create_model("legacy_xception", pretrained=pretrained, num_classes=0)
            return model
        except:
            # Fallback: create Xception manually if timm doesn't have it
            model = models.xception(pretrained=pretrained)
            # Remove the classifier
            model.fc = nn.Identity()
            return model
    
    def _create_efficientnet_b0(self, pretrained):
        """Create EfficientNet-B0 backbone."""
        try:
            model = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
            return model
        except:
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier = nn.Identity()
            return model
    
    def _create_efficientnet_b3(self, pretrained):
        """Create EfficientNet-B3 backbone."""
        try:
            model = timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=0)
            return model
        except:
            model = models.efficientnet_b3(pretrained=pretrained)
            model.classifier = nn.Identity()
            return model
    
    def _create_resnet50(self, pretrained):
        """Create ResNet-50 backbone."""
        model = models.resnet50(pretrained=pretrained)
        # Remove the final fully connected layer
        model.fc = nn.Identity()
        return model
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Handle different feature shapes
        if len(features.shape) == 4:
            # Spatial features: (batch, channels, height, width)
            # Apply full classifier pipeline
            x = self.classifier(features)
        elif len(features.shape) == 2:
            # Already flattened: (batch, features)
            # Skip pooling and flatten, go directly to classifier layers
            x = features
            # Apply dropout and linear layers (skip pooling and flatten)
            for layer in self.classifier[2:]:
                x = layer(x)
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        return x


def create_model(model_name=config.MODEL_NAME, pretrained=config.PRETRAINED, 
                 num_classes=config.NUM_CLASSES, dropout_rate=config.DROPOUT_RATE):
    """
    Factory function to create a model.
    
    Args:
        model_name: Name of backbone architecture
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes
        dropout_rate: Dropout rate
    
    Returns:
        Initialized model
    """
    model = DeepFakeDetector(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    return model


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

