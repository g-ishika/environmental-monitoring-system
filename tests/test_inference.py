"""Tests for inference module"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from inference.predictor import Predictor, PredictionResult
from inference.alert_system import AlertSystem, Alert


class TestPredictor:
    """Test predictor functionality"""
    
    @pytest.fixture
    def config(self):
        return {
            'threshold': 0.7,
            'classes': ['chainsaw', 'gunshot', 'vehicle', 'fire', 'animal_distress', 'background'],
            'sample_rate': 22050,
            'duration': 5.0,
            'n_mels': 128
        }
    
    @pytest.fixture
    def predictor(self, config, tmp_path):
        # Create a dummy model
        model_path = tmp_path / "dummy_model.pt"
        # Save dummy model
        dummy_model = torch.nn.Linear(10, 6)
        torch.save(dummy_model.state_dict(), model_path)
        
        return Predictor(str(model_path), config, device='cpu')
    
    def test_predictor_initialization(self, predictor):
        assert predictor is not None
        assert predictor.threshold == 0.7
        assert len(predictor.classes) == 6
    
    def test_preprocess_audio(self, predictor, tmp_path):
        # Create dummy audio file
        audio_path = tmp_path / "test.wav"
        dummy_audio = np.random.randn(22050 * 5).astype(np.float32)
        import soundfile as sf
        sf.write(audio_path, dummy_audio, 22050)
        
        mel_spec, features = predictor.preprocess_audio(str(audio_path))
        assert mel_spec.shape == (1, 3, 128, 128)
        assert 'mfcc_mean' in features


class TestAlertSystem:
    """Test alert system functionality"""
    
    @pytest.fixture
    def config(self):
        return {
            'alerting': {
                'channels': ['email'],
                'severity_levels': ['info', 'warning', 'critical'],
                'cooldown': 60
            }
        }
    
    @pytest.fixture
    def alert_system(self, config):
        return AlertSystem(config)
    
    def test_alert_creation(self, alert_system):
        result = {
            'class_name': 'chainsaw',
            'confidence': 0.95,
            'features': {'test': 1.0}
        }
        alert = alert_system.create_alert(result, 'Test Location')
        
        assert alert.event_type == 'chainsaw'
        assert alert.confidence == 0.95
        assert alert.severity == 'critical'
        assert alert.location == 'Test Location'
    
    def test_alert_severity(self, alert_system):
        assert alert_system.determine_severity(0.96) == 'critical'
        assert alert_system.determine_severity(0.88) == 'warning'
        assert alert_system.determine_severity(0.75) == 'info'
    
    def test_alert_cooldown(self, alert_system):
        # First alert should not be in cooldown
        assert alert_system.check_cooldown('chainsaw', 'Test') == False
        
        # Immediately after, should be in cooldown
        assert alert_system.check_cooldown('chainsaw', 'Test') == True