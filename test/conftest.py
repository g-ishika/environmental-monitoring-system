"""Pytest configuration"""

import pytest
import torch
from pathlib import Path

@pytest.fixture
def sample_audio():
    """Create a sample audio tensor"""
    return torch.randn(22050 * 5)  # 5 seconds at 22.05kHz

@pytest.fixture
def config_path():
    return Path("config/config.yaml")