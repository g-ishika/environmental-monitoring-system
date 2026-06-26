"""Training pipeline for environmental monitoring models"""

import torch
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from datetime import datetime

from data.dataset import create_dataloaders
from models.cnn_model import create_model
from training.trainer import ModelTrainer
from training.evaluator import ModelEvaluator
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TrainingPipeline:
    """Complete training pipeline for environmental audio classification"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.trainer = None
    
    def prepare_data(self, data_dir: str = "data/processed") -> Tuple:
        """Prepare data loaders"""
        logger.info("Preparing data...")
        
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=self.config['data']['batch_size'],
            train_split=self.config['data']['train_split'],
            val_split=self.config['data']['val_split'],
            test_split=self.config['data']['test_split'],
            num_workers=self.config['data'].get('num_workers', 4),
            class_names=self.config['data']['classes']
        )
        
        logger.info(f"Train: {len(self.train_loader.dataset)} samples")
        logger.info(f"Val: {len(self.val_loader.dataset)} samples")
        logger.info(f"Test: {len(self.test_loader.dataset)} samples")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def create_model(self) -> torch.nn.Module:
        """Create model"""
        logger.info("Creating model...")
        
        self.model = create_model(
            model_name=self.config['model']['architecture'],
            num_classes=self.config['model']['num_classes'],
            dropout_rate=self.config['model']['dropout_rate']
        )
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def train(self) -> Dict:
        """Run training"""
        # Prepare data if not already done
        if self.train_loader is None:
            self.prepare_data()
        
        # Create model if not already done
        if self.model is None:
            self.create_model()
        
        # Create trainer
        self.trainer = ModelTrainer(
            model=self.model,
            config=self.config['training'],
            device=self.device,
            use_wandb=self.config.get('use_wandb', False)
        )
        
        # Train
        logger.info("Starting training...")
        history = self.trainer.train(self.train_loader, self.val_loader)
        
        # Evaluate
        logger.info("Evaluating on test set...")
        metrics = self.trainer.evaluate(self.test_loader)
        
        # Save results
        self._save_results(history, metrics)
        
        return {
            'history': history,
            'metrics': metrics
        }
    
    def _save_results(self, history: Dict, metrics: Dict):
        """Save training results"""
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metrics
        with open(output_dir / f'metrics_{timestamp}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save config
        with open(output_dir / f'config_{timestamp}.yaml', 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Results saved to {output_dir}")
    
    def load_model(self, model_path: str):
        """Load saved model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        logger.info(f"Model loaded from {model_path}")
        
        return self.model
    
    def get_model_summary(self) -> str:
        """Get model summary"""
        if self.model is None:
            return "Model not created yet"
        
        from torchinfo import summary
        return str(summary(self.model, input_size=(1, 3, 128, 128)))