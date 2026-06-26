"""Tests for model definitions"""

import pytest
import torch
import numpy as np

from models.cnn_model import create_model, ResNet, CustomCNN, AttentionCNN
from models.ensemble import ModelEnsemble


class TestModels:
    """Test model creation and forward pass"""
    
    @pytest.fixture
    def input_tensor(self):
        return torch.randn(2, 3, 128, 128)
    
    def test_resnet18(self, input_tensor):
        model = create_model('resnet18', num_classes=6)
        output = model(input_tensor)
        
        assert output.shape == (2, 6)
        assert model.training == True
    
    def test_resnet50(self, input_tensor):
        model = create_model('resnet50', num_classes=6)
        output = model(input_tensor)
        
        assert output.shape == (2, 6)
    
    def test_custom_cnn(self, input_tensor):
        model = create_model('custom_cnn', num_classes=6)
        output = model(input_tensor)
        
        assert output.shape == (2, 6)
    
    def test_attention_cnn(self, input_tensor):
        model = create_model('attention_cnn', num_classes=6)
        output = model(input_tensor)
        
        assert output.shape == (2, 6)
    
    def test_ensemble(self, input_tensor):
        # Create multiple models
        models = [
            create_model('resnet18', num_classes=6),
            create_model('custom_cnn', num_classes=6)
        ]
        
        ensemble = ModelEnsemble(models)
        output = ensemble(input_tensor)
        
        assert output.shape == (2, 6)
    
    def test_model_parameters(self):
        model = create_model('resnet18', num_classes=6)
        total_params = sum(p.numel() for p in model.parameters())
        
        assert total_params > 0
        assert total_params < 20_000_000  # ResNet18 has ~11M parameters
    
    def test_model_train_mode(self):
        model = create_model('resnet50', num_classes=6)
        assert model.training == True
        
        model.eval()
        assert model.training == False
    
    def test_gradient_flow(self, input_tensor):
        model = create_model('resnet50', num_classes=6)
        output = model(input_tensor)
        
        loss = output.sum()
        loss.backward()
        
        # Check if gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None