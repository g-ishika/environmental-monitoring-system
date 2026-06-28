

import torch
import torchaudio
import numpy as np
import random


class DataAugmentation:
    
    
    def __init__(self, sample_rate: int = 22050, apply_prob: float = 0.5):
        self.sample_rate = sample_rate
        self.apply_prob = apply_prob
    
    def time_stretch(self, audio: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
        """Time stretch augmentation"""
        if rate == 1.0:
            return audio
        
        
        audio_np = audio.numpy()
        
        
        import librosa
        stretched = librosa.effects.time_stretch(audio_np, rate=rate)
        
        
        if len(stretched) < len(audio_np):
            stretched = np.pad(stretched, (0, len(audio_np) - len(stretched)))
        else:
            stretched = stretched[:len(audio_np)]
        
        return torch.from_numpy(stretched).float()
    
    def pitch_shift(self, audio: torch.Tensor, semitones: float = 0.0) -> torch.Tensor:
        """Pitch shift augmentation"""
        if semitones == 0:
            return audio
        
        # since we arr going to use lebrosa
        audio_np = audio.numpy()
        
        
        import librosa
        shifted = librosa.effects.pitch_shift(
            audio_np, sr=self.sample_rate, n_steps=semitones
        )
        
        return torch.from_numpy(shifted).float()
    
    def add_noise(self, audio: torch.Tensor, noise_level: float = 0.005) -> torch.Tensor:
        """Add random noise"""
        noise = torch.randn_like(audio) * noise_level
        return audio + noise
    
    def time_shift(self, audio: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
        """Time shift augmentation"""
        shift_samples = int(shift * self.sample_rate)
        return torch.roll(audio, shift_samples)
    
    def augment(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations"""
        
        if random.random() < self.apply_prob:
            rate = random.uniform(0.8, 1.2)
            audio = self.time_stretch(audio, rate)
        
        
        if random.random() < self.apply_prob:
            semitones = random.uniform(-2, 2)
            audio = self.pitch_shift(audio, semitones)
        
        
        if random.random() < self.apply_prob * 0.5:
            noise_level = random.uniform(0.001, 0.01)
            audio = self.add_noise(audio, noise_level)
        
        
        if random.random() < self.apply_prob * 0.3:
            shift = random.uniform(-0.5, 0.5)
            audio = self.time_shift(audio, shift)
        
        return audio