import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np


class ModelEnsemble(nn.Module):
    """Ensemble of multiple models for robust predictions"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            self.weights = weights / np.sum(weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble"""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            with torch.set_grad_enabled(False):
                pred = model(x)
                predictions.append(pred * weight)
        
        # average
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        return ensemble_pred
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict:
        """Get predictions with uncertainty estimates"""
        predictions = []
        for model in self.models:
            with torch.set_grad_enabled(False):
                pred = torch.softmax(model(x), dim=1)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # (num_models, batch_size, num_classes)
        
        mean_pred = torch.mean(predictions, dim=0)
        std_pred = torch.std(predictions, dim=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'confidence': 1 - std_pred.mean(dim=1)  # Higher std = lower confidence
        }


class WeightedEnsemble:
    """Weighted ensemble for inference"""
    
    def __init__(self, models: List[nn.Module], weights: List[float]):
        self.models = models
        self.weights = weights
        self.device = next(models[0].parameters()).device
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Run ensemble prediction"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = torch.softmax(model(x), dim=1)
                predictions.append(pred.cpu().numpy() * weight)
        
        # average
        ensemble_pred = np.sum(predictions, axis=0)
        return ensemble_pred
    
    def get_confidence(self, x: torch.Tensor) -> float:
        """Get prediction confidence"""
        pred = self.predict(x)
        return np.max(pred, axis=1)[0]


def create_ensemble(config: Dict) -> ModelEnsemble:
    """Create ensemble from configuration"""
    from .cnn_model import create_model
    
    model_names = config.get('ensemble_models', ['resnet18', 'resnet50', 'attention_cnn'])
    num_classes = config.get('num_classes', 6)
    dropout_rate = config.get('dropout_rate', 0.3)
    
    models = []
    for name in model_names:
        model = create_model(name, num_classes, dropout_rate)
        models.append(model)
    
    weights = config.get('ensemble_weights', None)
    return ModelEnsemble(models, weights)