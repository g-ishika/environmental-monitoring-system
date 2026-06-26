"""PyTorch dataset for environmental audio"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json
import random
from tqdm import tqdm
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .preprocessing import AudioPreprocessor
from .augmentation import DataAugmentation
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioDataset(Dataset):
    """PyTorch dataset for environmental audio classification"""
    
    def __init__(
        self,
        raw_dir: str = "audio_data/raw",
        processed_dir: str = "audio_data/processed",
        class_names: List[str] = None,
        preprocessor: AudioPreprocessor = None,
        augmenter: Optional[DataAugmentation] = None,
        transform: Optional[callable] = None,
        cache_data: bool = False,
        use_processed: bool = False
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.use_processed = use_processed
        
        if class_names is None:
            class_names = ['chainsaw', 'gunshot', 'vehicle', 'fire', 'animal_distress', 'background']
        
        self.class_names = class_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.augmenter = augmenter
        self.transform = transform
        self.cache_data = cache_data
        
        # Find all audio files
        self.file_paths = []
        self.labels = []
        self.cached_data = {} if cache_data else None
        
        if use_processed and self.processed_dir.exists():
            self._load_processed_data()
        else:
            self._scan_raw_files()
        
        logger.info(f"Found {len(self.file_paths)} audio files across {len(class_names)} classes")
    
    def _load_processed_data(self):
        """Load preprocessed data from processed directory"""
        try:
            data = torch.load(self.processed_dir / 'processed_data.pt')
            self.file_paths = data['file_paths']
            self.labels = data['labels']
            self.cached_data = data.get('mel_spectrograms', {})
            logger.info(f"Loaded {len(self.file_paths)} processed samples")
        except Exception as e:
            logger.warning(f"Could not load processed data: {e}, falling back to raw")
            self._scan_raw_files()
    
    def _scan_raw_files(self):
        """Scan raw directory for audio files"""
        for class_name in self.class_names:
            class_dir = self.raw_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
            
            # Find audio files
            audio_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3"))
            audio_files += list(class_dir.glob("*.flac")) + list(class_dir.glob("*.ogg"))
            
            for file_path in audio_files:
                self.file_paths.append(file_path)
                self.labels.append(self.class_to_idx[class_name])
        
        # Shuffle
        combined = list(zip(self.file_paths, self.labels))
        random.shuffle(combined)
        self.file_paths, self.labels = zip(*combined)
        self.file_paths = list(self.file_paths)
        self.labels = list(self.labels)
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample"""
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Check cache
        if self.cached_data and str(file_path) in self.cached_data:
            mel_spec = self.cached_data[str(file_path)]
        else:
            # Load and process audio
            audio = self.preprocessor.load_audio(str(file_path))
            
            # Apply augmentation if available
            if self.augmenter and self.transform is None:
                audio = self.augmenter.augment(audio)
            
            # Process audio
            processed = self.preprocessor.process_audio(audio, apply_noise_reduction=True)
            mel_spec = processed['mel_spectrogram']
            
            # Convert to 3-channel image
            mel_spec = mel_spec.unsqueeze(0)  # Add channel dimension
            mel_spec = mel_spec.repeat(3, 1, 1)  # Repeat for 3 channels
            
            # Cache if enabled
            if self.cached_data is not None:
                self.cached_data[str(file_path)] = mel_spec
        
        # Apply transform
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        return mel_spec, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets"""
        labels = np.array(self.labels)
        class_counts = np.bincount(labels)
        total = len(labels)
        class_weights = total / (len(class_counts) * class_counts)
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        return class_weights
    
    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        labels = np.array(self.labels)
        class_counts = np.bincount(labels)
        
        return {
            'total_samples': len(self.file_paths),
            'num_classes': len(self.class_names),
            'class_counts': {self.class_names[i]: int(count) for i, count in enumerate(class_counts)},
            'class_names': self.class_names
        }


def create_dataloaders(
    data_dir: str = "audio_data/processed",
    raw_dir: str = "audio_data/raw",
    batch_size: int = 32,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    class_names: Optional[List[str]] = None,
    augment_train: bool = True,
    device: str = "cpu",
    use_processed: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    # Default classes
    if class_names is None:
        class_names = [
            "chainsaw", "gunshot", "vehicle", 
            "fire", "animal_distress", "background"
        ]
    
    # Create preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        duration=5.0,
        n_mels=128,
        device=device
    )
    
    # Create augmenter for training
    augmenter = DataAugmentation() if augment_train else None
    
    # Create full dataset
    full_dataset = AudioDataset(
        raw_dir=raw_dir,
        processed_dir=data_dir,
        class_names=class_names,
        preprocessor=preprocessor,
        augmenter=augmenter,
        use_processed=use_processed
    )
    
    # Calculate split sizes
    total = len(full_dataset)
    train_size = int(train_split * total)
    val_size = int(val_split * total)
    test_size = total - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size]
    )
    
    logger.info(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader