"""Custom dataloader implementations"""

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Iterator, List, Optional, Tuple
import numpy as np
import random


class BalancedBatchSampler(Sampler):
    """Balanced batch sampler for imbalanced datasets"""
    
    def __init__(self, labels: List[int], batch_size: int, drop_last: bool = False):
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        # Get number of classes
        self.num_classes = len(self.class_indices)
        
        # Calculate samples per class per batch
        self.samples_per_class = batch_size // self.num_classes
        if self.samples_per_class == 0:
            self.samples_per_class = 1
        
        # Total batches
        self.total_samples = len(labels)
        self.num_batches = self.total_samples // batch_size
        
        if not self.drop_last and self.total_samples % batch_size != 0:
            self.num_batches += 1
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each class
        class_shuffled = {}
        for class_id, indices in self.class_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            class_shuffled[class_id] = shuffled
        
        # Create batches
        batch_indices = []
        
        for batch_idx in range(self.num_batches):
            batch = []
            
            for class_id in range(self.num_classes):
                class_indices = class_shuffled.get(class_id, [])
                if not class_indices:
                    continue
                
                # Take samples from this class
                samples = min(self.samples_per_class, len(class_indices))
                batch.extend(class_indices[:samples])
                
                # Rotate indices
                class_shuffled[class_id] = class_indices[samples:] + class_indices[:samples]
            
            # Shuffle batch
            random.shuffle(batch)
            
            # Truncate if needed
            if len(batch) > self.batch_size:
                batch = batch[:self.batch_size]
            
            batch_indices.append(batch)
        
        return iter(batch_indices)
    
    def __len__(self) -> int:
        return self.num_batches


class WeightedRandomSampler(Sampler):
    """Weighted random sampler for imbalanced datasets"""
    
    def __init__(self, weights: List[float], num_samples: int, replacement: bool = True):
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
    
    def __iter__(self) -> Iterator[int]:
        return iter(torch.multinomial(
            torch.tensor(self.weights, dtype=torch.float64),
            self.num_samples,
            replacement=self.replacement
        ).tolist())
    
    def __len__(self) -> int:
        return self.num_samples


class AudioDataLoader(DataLoader):
    """Custom dataloader with additional features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_batch_stats(self):
        """Get statistics about the current batch"""
        if not self.dataset:
            return {}
        
        # Get one batch to analyze
        batch = next(iter(self))
        if isinstance(batch, tuple):
            data, labels = batch
        else:
            data = batch
            labels = None
        
        stats = {
            'batch_size': data.size(0),
            'shape': data.shape,
            'dtype': data.dtype,
            'device': data.device,
            'min_val': data.min().item(),
            'max_val': data.max().item(),
            'mean': data.mean().item(),
            'std': data.std().item()
        }
        
        if labels is not None:
            stats['num_classes'] = len(torch.unique(labels))
            stats['class_distribution'] = torch.bincount(labels).tolist()
        
        return stats