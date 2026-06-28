"""Training callbacks for PyTorch"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import json
from datetime import datetime
import matplotlib.pyplot as plt


class Callback:
    
    
    def on_train_begin(self, trainer):
        pass
    
    def on_train_end(self, trainer):
        pass
    
    def on_epoch_begin(self, trainer, epoch):
        pass
    
    def on_epoch_end(self, trainer, epoch, metrics):
        pass
    
    def on_batch_begin(self, trainer, batch):
        pass
    
    def on_batch_end(self, trainer, batch, loss):
        pass


class ModelCheckpoint(Callback):
    """Save model checkpoints"""
    
    def __init__(self, save_dir: str = "checkpoints", save_best: bool = True, 
                 save_last: bool = True, metric: str = 'val_acc', mode: str = 'max'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.save_best = save_best
        self.save_last = save_last
        self.metric = metric
        self.mode = mode
        self.best_value = -float('inf') if mode == 'max' else float('inf')
        self.best_model_state = None
    
    def on_epoch_end(self, trainer, epoch, metrics):
        # Save last checkpoint
        if self.save_last:
            checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': metrics
            }, checkpoint_path)
        
        # Save best checkpoint
        if self.save_best and self.metric in metrics:
            current_value = metrics[self.metric]
            is_better = (self.mode == 'max' and current_value > self.best_value) or \
                       (self.mode == 'min' and current_value < self.best_value)
            
            if is_better:
                self.best_value = current_value
                self.best_model_state = trainer.model.state_dict().copy()
                
                best_path = self.save_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_model_state,
                    'metrics': metrics
                }, best_path)


class EarlyStopping(Callback):
    """Early stopping callback"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 metric: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.should_stop = False
    
    def on_epoch_end(self, trainer, epoch, metrics):
        if self.metric not in metrics:
            return
        
        current_value = metrics[self.metric]
        is_better = (self.mode == 'min' and current_value < self.best_value - self.min_delta) or \
                   (self.mode == 'max' and current_value > self.best_value + self.min_delta)
        
        if is_better:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback"""
    
    def __init__(self, scheduler, monitor: str = 'val_loss'):
        self.scheduler = scheduler
        self.monitor = monitor
    
    def on_epoch_end(self, trainer, epoch, metrics):
        if hasattr(self.scheduler, 'step') and self.monitor in metrics:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics[self.monitor])
            else:
                self.scheduler.step()


class TensorBoardLogger(Callback):
    """TensorBoard logging callback"""
    
    def __init__(self, log_dir: str = "logs"):
        from torch.utils.tensorboard import SummaryWriter
        self.log_dir = Path(log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(str(self.log_dir))
    
    def on_train_begin(self, trainer):
        # Log model graph
        dummy_input = torch.randn(1, 3, 128, 128).to(trainer.device)
        self.writer.add_graph(trainer.model, dummy_input)
    
    def on_epoch_end(self, trainer, epoch, metrics):
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, epoch)
        
        # Log learning rate
        lr = trainer.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, epoch)
    
    def on_train_end(self, trainer):
        self.writer.close()


class Callbacks:
    """Collection of callbacks"""
    
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks
    
    def on_train_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer):
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, trainer, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(self, trainer, epoch, metrics):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, metrics)
    
    def on_batch_begin(self, trainer, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(trainer, batch)
    
    def on_batch_end(self, trainer, batch, loss):
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch, loss)