import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Advanced model evaluation for PyTorch"""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
    def evaluate(self, test_loader, class_names: List[str]) -> Dict:
        """Comprehensive model evaluation"""
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Compute all metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, average='weighted'),
            'recall': recall_score(all_targets, all_preds, average='weighted'),
            'f1_macro': f1_score(all_targets, all_preds, average='macro'),
            'f1_weighted': f1_score(all_targets, all_preds, average='weighted'),
        }
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                'precision': precision_score(all_targets, all_preds, labels=[i], average=None)[0],
                'recall': recall_score(all_targets, all_preds, labels=[i], average=None)[0],
                'f1': f1_score(all_targets, all_preds, labels=[i], average=None)[0]
            }
        
        metrics['per_class'] = per_class_metrics
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(all_targets, all_preds)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            all_targets, all_preds,
            target_names=class_names,
            output_dict=True
        )
        
        return metrics
    
    def plot_confusion_matrix(self, metrics: Dict, class_names: List[str], 
                             save_path: Optional[str] = None):
        """Plot confusion matrix"""
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues',
            cbar=True
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_model_summary(self, input_shape: Tuple[int, ...]) -> str:
        """Get model summary"""
        from torchinfo import summary
        return str(summary(self.model, input_size=(1, *input_shape)))


class EarlyStopping:
    """Early stopping for PyTorch"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop