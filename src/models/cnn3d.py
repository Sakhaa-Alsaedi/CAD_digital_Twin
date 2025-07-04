"""
Enhanced 3D CNN models for echocardiogram analysis.

This module provides improved 3D CNN architectures with attention mechanisms,
residual connections, and better regularization for medical video analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for 3D CNNs."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.sigmoid(self.conv(x))
        return x * attention


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for video sequences."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.temporal_conv = nn.Conv3d(in_channels, in_channels, 
                                     kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.attention_conv = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_features = self.temporal_conv(x)
        attention_weights = self.sigmoid(self.attention_conv(temporal_features))
        return x * attention_weights


class ResidualBlock3D(nn.Module):
    """3D Residual block with optional attention."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 use_attention: bool = False):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        self.attention = SpatialAttention(out_channels) if use_attention else None
        self.dropout = nn.Dropout3d(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        if self.attention:
            out = self.attention(out)
        
        out += residual
        out = F.relu(out)
        
        return out


class Enhanced3DCNN(nn.Module):
    """
    Enhanced 3D CNN for echocardiogram analysis with attention mechanisms
    and improved architecture.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 1,
                 base_channels: int = 64,
                 use_attention: bool = True,
                 dropout_rate: float = 0.2):
        """
        Initialize Enhanced 3D CNN.
        
        Args:
            input_channels: Number of input channels (typically 3 for RGB)
            num_classes: Number of output classes/parameters
            base_channels: Base number of channels for the first layer
            use_attention: Whether to use attention mechanisms
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.use_attention = use_attention
        
        # Initial convolution
        self.conv1 = nn.Conv3d(input_channels, base_channels, kernel_size=7, 
                              stride=(1, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Residual layers
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=(1, 1, 1))
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=(2, 2, 2))
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=(2, 2, 2))
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=(2, 2, 2))
        
        # Attention mechanisms
        if use_attention:
            self.spatial_attention = SpatialAttention(base_channels * 8)
            self.temporal_attention = TemporalAttention(base_channels * 8)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(base_channels * 8, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Enhanced3DCNN initialized with {self._count_parameters()} parameters")
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: Tuple[int, int, int]) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock3D(in_channels, out_channels, stride, self.use_attention))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, 
                                        use_attention=self.use_attention))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.spatial_attention(x)
            x = self.temporal_attention(x)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class ResNet3D(nn.Module):
    """
    3D ResNet architecture for medical video analysis.
    """
    
    def __init__(self, 
                 layers: List[int] = [2, 2, 2, 2],
                 input_channels: int = 3,
                 num_classes: int = 1,
                 base_channels: int = 64):
        """
        Initialize 3D ResNet.
        
        Args:
            layers: Number of blocks in each layer
            input_channels: Number of input channels
            num_classes: Number of output classes
            base_channels: Base number of channels
        """
        super().__init__()
        
        self.base_channels = base_channels
        
        # Initial layers
        self.conv1 = nn.Conv3d(input_channels, base_channels, kernel_size=7, 
                              stride=(1, 2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # Residual layers
        self.layer1 = self._make_layer(base_channels, layers[0], stride=(1, 1, 1))
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=(2, 2, 2))
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=(2, 2, 2))
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=(2, 2, 2))
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(base_channels * 8, num_classes)
        
        self._initialize_weights()
        
        logger.info(f"ResNet3D initialized with {self._count_parameters()} parameters")
    
    def _make_layer(self, out_channels: int, num_blocks: int, 
                   stride: Tuple[int, int, int]) -> nn.Sequential:
        """Create a residual layer."""
        layers = []
        layers.append(ResidualBlock3D(self.base_channels, out_channels, stride))
        self.base_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class MultiTaskCNN3D(nn.Module):
    """
    Multi-task 3D CNN for simultaneous parameter prediction and classification.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_regression_outputs: int = 7,
                 num_classification_outputs: int = 2,
                 base_channels: int = 64):
        """
        Initialize multi-task 3D CNN.
        
        Args:
            input_channels: Number of input channels
            num_regression_outputs: Number of regression outputs (parameters)
            num_classification_outputs: Number of classification outputs
            base_channels: Base number of channels
        """
        super().__init__()
        
        # Shared backbone
        self.backbone = Enhanced3DCNN(
            input_channels=input_channels,
            num_classes=base_channels * 8,  # Feature dimension
            base_channels=base_channels,
            use_attention=True
        )
        
        # Remove the final FC layer from backbone
        self.backbone.fc = nn.Identity()
        
        # Task-specific heads
        self.regression_head = nn.Sequential(
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(base_channels * 4, num_regression_outputs)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(base_channels * 4, num_classification_outputs)
        )
        
        logger.info(f"MultiTaskCNN3D initialized with {self._count_parameters()} parameters")
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-task learning.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (regression_output, classification_output)
        """
        # Extract features using shared backbone
        features = self.backbone(x)
        
        # Task-specific predictions
        regression_output = self.regression_head(features)
        classification_output = self.classification_head(features)
        
        return regression_output, classification_output

