import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import wandb

from utils.logger import setup_logger
from utils.metrics import MetricsTracker

logger = setup_logger(__name__)


class ModelTrainer:
    """Advanced PyTorch model trainer"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = 'cuda',
        use_wandb: bool = False
    ):
        self.model = model
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_loss()
        
        # Mixed precision training
        self.use_mixed_precision = config.get('use_mixed_precision', True)
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Metrics
        self.metrics = MetricsTracker()
        self.best_val_acc = 0.0
        self.best_model_state = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Setup directories
        self.checkpoint_dir = Path('models_checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logs_dir = Path('logs') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        if use_wandb:
            wandb.init(
                project="environmental-monitoring",
                config=config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0001)
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_name = self.config.get('scheduler', 'cosine')
        epochs = self.config.get('epochs', 100)
        
        if scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=1e-7
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        else:
            return None
    
    def _create_loss(self) -> nn.Module:
        """Create loss function"""
        label_smoothing = self.config.get('label_smoothing', 0.0)
        
        if label_smoothing > 0:
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            return nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader: DataLoader) -> Dict:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc='Validating'):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate additional metrics
        metrics = self.metrics.compute(all_targets, all_preds)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total,
            **metrics
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Complete training pipeline"""
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('early_stopping_patience', 10)
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\nEpoch {epoch}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['learning_rates'].append(current_lr)
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy'],
                    'learning_rate': current_lr,
                    **{f'val_{k}': v for k, v in val_metrics.items() if k not in ['loss', 'accuracy']}
                })
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(epoch, val_metrics)
                patience_counter = 0
                logger.info(f"New best model! Accuracy: {self.best_val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Print epoch summary
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'metrics': metrics,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save best model separately
        best_path = self.checkpoint_dir / 'best_model.pt'
        torch.save({
            'model_state_dict': self.best_model_state,
            'metrics': metrics,
            'config': self.config
        }, best_path)
    
    def save_history(self):
        """Save training history"""
        history_path = self.logs_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Also save as numpy for easy loading
        np.savez(
            self.logs_dir / 'history.npz',
            train_loss=np.array(self.history['train_loss']),
            train_acc=np.array(self.history['train_acc']),
            val_loss=np.array(self.history['val_loss']),
            val_acc=np.array(self.history['val_acc']),
            learning_rates=np.array(self.history['learning_rates'])
        )
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(test_loader, desc='Testing'):
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Compute all metrics
        metrics = self.metrics.compute_all(all_targets, all_preds)
        
        # Log results
        logger.info("Test Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics