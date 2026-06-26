"""Tests for preprocessing module"""

import pytest
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

from data.preprocessing import AudioPreprocessor
from data.augmentation import DataAugmentation


class TestAudioPreprocessor:
    """Test audio preprocessing"""
    
    @pytest.fixture
    def preprocessor(self):
        return AudioPreprocessor(
            sample_rate=22050,
            duration=2.0,
            n_mels=64
        )
    
    @pytest.fixture
    def dummy_audio(self):
        # Create dummy audio signal
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        return torch.from_numpy(audio).float()
    
    def test_load_audio(self, preprocessor, tmp_path):
        # Create dummy audio file
        audio_path = tmp_path / "test.wav"
        dummy_audio = np.random.randn(22050 * 2).astype(np.float32)
        sf.write(audio_path, dummy_audio, 22050)
        
        audio = preprocessor.load_audio(str(audio_path))
        assert len(audio) == preprocessor.n_samples
        assert isinstance(audio, torch.Tensor)
    
    def test_extract_mel_spectrogram(self, preprocessor, dummy_audio):
        mel_spec = preprocessor.extract_mel_spectrogram(dummy_audio)
        
        assert mel_spec.shape == (64, 87)  # (n_mels, time_frames)
        assert torch.min(mel_spec) >= 0
        assert torch.max(mel_spec) <= 1
    
    def test_extract_features(self, preprocessor, dummy_audio):
        features = preprocessor.extract_features(dummy_audio)
        
        assert 'mfcc_mean' in features
        assert 'spectral_centroid' in features
        assert 'zcr' in features
        assert 'rms' in features
    
    def test_noise_reduction(self, preprocessor, dummy_audio):
        # Add noise
        noisy_audio = dummy_audio + torch.randn_like(dummy_audio) * 0.1
        
        # Apply noise reduction
        cleaned_audio = preprocessor.apply_noise_reduction(noisy_audio)
        
        assert len(cleaned_audio) == len(noisy_audio)
        assert torch.std(cleaned_audio) < torch.std(noisy_audio)


class TestDataAugmentation:
    """Test data augmentation"""
    
    @pytest.fixture
    def augmenter(self):
        return DataAugmentation(sample_rate=22050, apply_prob=1.0)
    
    @pytest.fixture
    def dummy_audio(self):
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)
        return torch.from_numpy(audio).float()
    
    def test_time_stretch(self, augmenter, dummy_audio):
        stretched = augmenter.time_stretch(dummy_audio, 1.5)
        assert len(stretched) == len(dummy_audio)  # Should maintain length
    
    def test_pitch_shift(self, augmenter, dummy_audio):
        shifted = augmenter.pitch_shift(dummy_audio, 2.0)
        assert len(shifted) == len(dummy_audio)
    
    def test_add_noise(self, augmenter, dummy_audio):
        noisy = augmenter.add_noise(dummy_audio, 0.01)
        assert len(noisy) == len(dummy_audio)
        assert torch.std(noisy) > torch.std(dummy_audio)
    
    def test_time_shift(self, augmenter, dummy_audio):
        shifted = augmenter.time_shift(dummy_audio, 0.5)
        assert len(shifted) == len(dummy_audio)
        assert not torch.allclose(shifted, dummy_audio)
    
    def test_full_augmentation(self, augmenter, dummy_audio):
        augmented = augmenter.augment(dummy_audio)
        assert len(augmented) == len(dummy_audio)
        assert isinstance(augmented, torch.Tensor)