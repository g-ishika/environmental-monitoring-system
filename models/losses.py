"""Custom loss functions for environmental audio classification"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss"""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed labels
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        targets_smoothed = (1 - self.smoothing) * targets_one_hot + self.smoothing / self.num_classes
        
        loss = -(targets_smoothed * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """Combined loss function with multiple components"""
    
    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 0.7,
        focal_weight: float = 0.3,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        
        self.ce_loss = LabelSmoothingLoss(num_classes, label_smoothing)
        self.focal_loss = FocalLoss(focal_alpha, focal_gamma)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        
        return self.ce_weight * ce_loss + self.focal_weight * focal_loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning discriminative features"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()
        negative_mask = 1 - positive_mask
        
        # Compute contrastive loss
        exp_similarity = torch.exp(similarity)
        positive_sum = (exp_similarity * positive_mask).sum(dim=1)
        negative_sum = (exp_similarity * negative_mask).sum(dim=1)
        
        loss = -torch.log(positive_sum / (positive_sum + negative_sum))
        
        return loss.mean()


def create_loss(loss_type: str = 'cross_entropy', **kwargs) -> nn.Module:
    """Factory function to create loss"""
    
    loss_map = {
        'cross_entropy': nn.CrossEntropyLoss,
        'focal': FocalLoss,
        'label_smoothing': LabelSmoothingLoss,
        'combined': CombinedLoss,
        'contrastive': ContrastiveLoss
    }
    
    if loss_type not in loss_map:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss_map[loss_type](**kwargs)