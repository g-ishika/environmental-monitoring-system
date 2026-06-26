"""Attention-based models for environmental audio classification"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avgpool(x))
        max_out = self.fc(self.maxpool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class AttentionBlock(nn.Module):
    """General attention block with multiple attention mechanisms"""
    
    def __init__(self, channels: int, attention_type: str = 'se'):
        super().__init__()
        self.attention_type = attention_type
        
        if attention_type == 'se':
            self.attention = SEBlock(channels)
        elif attention_type == 'cbam':
            self.attention = CBAM(channels)
        elif attention_type == 'eca':
            self.attention = ECA(channels)
        else:
            self.attention = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ECA(nn.Module):
    """Efficient Channel Attention"""
    
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((channels.bit_length() - 1) / gamma) + b)
        kernel_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avgpool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y


class SelfAttention(nn.Module):
    """Self-Attention module for feature refinement"""
    
    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.query = nn.Conv2d(in_channels, out_channels, 1)
        self.key = nn.Conv2d(in_channels, out_channels, 1)
        self.value = nn.Conv2d(in_channels, out_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        
        # Query, Key, Value
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h * w)
        v = self.value(x).view(b, -1, h * w)
        
        # Attention scores
        attn = torch.bmm(q, k) / (c ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        
        return self.gamma * out + x


class AttentionResNetBlock(nn.Module):
    """ResNet block with attention mechanism"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        attention_type: str = 'se'
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.attention = AttentionBlock(out_channels, attention_type)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AttentionResNet(nn.Module):
    """ResNet with attention mechanisms"""
    
    def __init__(
        self,
        num_classes: int = 6,
        attention_type: str = 'se',
        dropout_rate: float = 0.3,
        input_channels: int = 3
    ):
        super().__init__()
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 2, 1, attention_type)
        self.layer2 = self._make_layer(128, 2, 2, attention_type)
        self.layer3 = self._make_layer(256, 2, 2, attention_type)
        self.layer4 = self._make_layer(512, 2, 2, attention_type)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
        attention_type: str
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(
                AttentionResNetBlock(
                    self.in_channels,
                    out_channels,
                    stride,
                    attention_type
                )
            )
            self.in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def create_attention_model(
    model_type: str = 'resnet',
    attention_type: str = 'se',
    num_classes: int = 6,
    dropout_rate: float = 0.3
) -> nn.Module:
    """Create attention-based model"""
    
    if model_type == 'resnet':
        return AttentionResNet(num_classes, attention_type, dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")