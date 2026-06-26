import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from typing import Dict, List, Optional


class MetricsTracker:
    """Track and compute various metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss = 0
        self.correct = 0
        self.total = 0
        self.preds = []
        self.targets = []
    
    def update(self, outputs, targets):
        """Update metrics with batch results"""
        _, predicted = outputs.max(1)
        self.preds.extend(predicted.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        
        self.total += targets.size(0)
        self.correct += predicted.eq(targets).sum().item()
    
    def compute_accuracy(self) -> float:
        """Compute accuracy"""
        return self.correct / self.total if self.total > 0 else 0
    
    def compute(self, targets=None, preds=None) -> Dict:
        """Compute all metrics"""
        if targets is None:
            targets = self.targets
        if preds is None:
            preds = self.preds
        
        if len(targets) == 0:
            return {}
        
        return {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, average='weighted', zero_division=0),
            'recall': recall_score(targets, preds, average='weighted', zero_division=0),
            'f1': f1_score(targets, preds, average='weighted', zero_division=0)
        }
    
    def compute_all(self, targets: List, preds: List) -> Dict:
        """Compute comprehensive metrics"""
        if len(targets) == 0:
            return {}
        
        targets = np.array(targets)
        preds = np.array(preds)
        
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision_macro': precision_score(targets, preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(targets, preds, average='weighted', zero_division=0),
            'recall_macro': recall_score(targets, preds, average='macro', zero_division=0),
            'recall_weighted': recall_score(targets, preds, average='weighted', zero_division=0),
            'f1_macro': f1_score(targets, preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(targets, preds, average='weighted', zero_division=0),
        }
        
        return metrics
    
    def get_confusion_matrix(self, targets: List, preds: List) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(targets, preds)