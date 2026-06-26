#!/usr/bin/env python
"""Train the environmental monitoring model"""

import argparse
import yaml
import torch
from pathlib import Path
import sys
import json
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import create_dataloaders
from models.cnn_model import create_model
from training.trainer import ModelTrainer
from training.evaluator import ModelEvaluator
from utils.logger import setup_logger

logger = setup_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train environmental monitoring model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='audio_data/processed',
        help='Path to processed data directory'
    )
    parser.add_argument(
        '--raw_dir',  # ← ADDED THIS!
        type=str,
        default='audio_data/raw',
        help='Path to raw audio data directory'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable wandb logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
        logger.info(f"Overriding epochs: {args.epochs}")
    
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
        logger.info(f"Overriding batch_size: {args.batch_size}")
    
    if args.no_wandb:
        config['use_wandb'] = False
        logger.info("wandb disabled")
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    logger.info(f"Using device: {args.device}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Raw data directory: {args.raw_dir}")
    
    # Create data loaders
    logger.info("=" * 50)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 50)
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            raw_dir=args.raw_dir,
            batch_size=config['data']['batch_size'],
            train_split=config['data']['train_split'],
            val_split=config['data']['val_split'],
            test_split=config['data']['test_split'],
            num_workers=config['data'].get('num_workers', 4),
            class_names=config['data']['classes'],
            device=args.device
        )
        
        logger.info(f"✅ Train samples: {len(train_loader.dataset)}")
        logger.info(f"✅ Validation samples: {len(val_loader.dataset)}")
        logger.info(f"✅ Test samples: {len(test_loader.dataset)}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        logger.error("Make sure you have audio files in audio_data/raw/")
        logger.error("Run: python scripts/download_data.py --dataset sample")
        sys.exit(1)
    
    # Create model
    logger.info("=" * 50)
    logger.info("STEP 2: Creating Model")
    logger.info("=" * 50)
    
    try:
        model = create_model(
            model_name=config['model']['architecture'],
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model']['dropout_rate']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model: {config['model']['architecture']}")
        logger.info(f"✅ Total parameters: {total_params:,}")
        logger.info(f"✅ Trainable parameters: {trainable_params:,}")
        logger.info(f"✅ Number of classes: {config['model']['num_classes']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create model: {e}")
        sys.exit(1)
    
    # Create trainer
    logger.info("=" * 50)
    logger.info("STEP 3: Setting Up Training")
    logger.info("=" * 50)
    
    try:
        trainer = ModelTrainer(
            model=model,
            config=config['training'],
            device=args.device,
            use_wandb=config.get('use_wandb', False)
        )
        
        logger.info(f"✅ Optimizer: {config['training'].get('optimizer', 'adam')}")
        logger.info(f"✅ Learning rate: {config['training'].get('learning_rate', 0.001)}")
        logger.info(f"✅ Epochs: {config['training'].get('epochs', 100)}")
        logger.info(f"✅ Batch size: {config['data']['batch_size']}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create trainer: {e}")
        sys.exit(1)
    
    # Train model
    logger.info("=" * 50)
    logger.info("STEP 4: Training")
    logger.info("=" * 50)
    
    try:
        history = trainer.train(train_loader, val_loader)
        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)
    
    # Evaluate on test set
    logger.info("=" * 50)
    logger.info("STEP 5: Evaluating Model")
    logger.info("=" * 50)
    
    try:
        evaluator = ModelEvaluator(model, device=args.device)
        metrics = evaluator.evaluate(test_loader, config['data']['classes'])
        
        logger.info("📊 Test Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        logger.info(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        sys.exit(1)
    
    # Save results
    logger.info("=" * 50)
    logger.info("STEP 6: Saving Results")
    logger.info("=" * 50)
    
    try:
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        
        metrics_path = output_dir / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✅ Metrics saved to: {metrics_path}")
        
        confusion_path = output_dir / 'confusion_matrix.png'
        evaluator.plot_confusion_matrix(
            metrics,
            config['data']['classes'],
            save_path=str(confusion_path)
        )
        logger.info(f"✅ Confusion matrix saved to: {confusion_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
    
    # Final summary
    logger.info("=" * 50)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("=" * 50)
    logger.info(f"Best model saved to: models_checkpoints/best_model.pt")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_macro']:.4f}")
    logger.info("=" * 50)
    
    return history, metrics


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)