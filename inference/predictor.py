"""Inference engine for environmental monitoring using PyTorch"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time
import json 
from dataclasses import dataclass
import yaml

from data.preprocessing import AudioPreprocessor
from models.cnn_model import create_model
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PredictionResult:
    """Prediction result container"""
    class_name: str
    confidence: float
    class_id: int
    timestamp: float
    features: Dict
    is_alert: bool = False
    all_probs: Optional[np.ndarray] = None


class Predictor:
    """Inference engine for environmental monitoring using PyTorch"""
    
    def __init__(self, model_path: str, config: Dict, device: str = 'cuda'):
        self.config = config
        self.threshold = config.get('threshold', 0.7)
        
        
        self.classes = config.get('classes', [])
        if not self.classes:
            
            self.classes = config.get('data', {}).get('classes', [])
        
        
        if not self.classes:
            self.classes = ['chainsaw', 'gunshot', 'vehicle', 'fire', 'animal_distress', 'background']
            logger.warning(f"No classes in config, using defaults: {self.classes}")
        
        self.num_classes = len(self.classes)
        
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Classes: {self.classes}")
        
        
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=config.get('sample_rate', 22050),
            duration=config.get('duration', 5.0),
            n_mels=config.get('n_mels', 128)
        )
        
        # Performance 
        self.inference_times = []
        self.total_predictions = 0
        
        # Alert cooldown
        self.last_alert_time = {}
        self.alert_cooldown = config.get('alert_cooldown', 60)
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained PyTorch model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            
            model_name = self.config.get('model', {}).get('architecture', 'resnet50')
            
            
            num_classes = self.num_classes
            
            logger.info(f"Creating model with {num_classes} classes")
            
            model = create_model(
                model_name=model_name,
                num_classes=num_classes,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.3)
            )
            
            # Load 
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model = model.to(self.device)
            logger.info(f"Loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, Dict]:
        """Preprocess audio for inference"""
        processed = self.preprocessor.process_audio(audio_path)
        mel_spec = processed['mel_spectrogram']
        
        
        mel_spec = mel_spec.unsqueeze(0)  # channel dimension
        mel_spec = mel_spec.repeat(3, 1, 1)  # for 3 channels
        mel_spec = mel_spec.unsqueeze(0)  
        
        return mel_spec.to(self.device), processed['features']
    
    def predict(self, audio_path: str) -> PredictionResult:
        """Run inference on audio file"""
        start_time = time.time()
        
        
        mel_spec, features = self.preprocess_audio(audio_path)
        
        
        with torch.no_grad():
            outputs = self.model(mel_spec)
            probabilities = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probabilities, dim=1)
        
        # Process predictions
        class_id = pred_class.item()
        confidence = probabilities[0, class_id].item()
        class_name = self.classes[class_id] if class_id < len(self.classes) else "unknown"
        
        
        is_alert = confidence >= self.threshold
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.total_predictions += 1
        
        result = PredictionResult(
            class_name=class_name,
            confidence=float(confidence),
            class_id=class_id,
            timestamp=time.time(),
            features=features,
            is_alert=is_alert,
            all_probs=probabilities.cpu().numpy()[0]
        )
        
        logger.debug(f"Prediction: {class_name} ({confidence:.2%}) in {inference_time:.3f}s")
        return result
    
    def predict_batch(self, audio_paths: List[str]) -> List[PredictionResult]:
        """Batch prediction for multiple audio files"""
        results = []
        
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                results.append(None)
        
        return results
    
    def predict_stream(self, audio_chunk: np.ndarray) -> PredictionResult:
        """Predict on streaming audio chunk"""
        # Process audio chunk
        processed = self.preprocessor.process_audio(audio_chunk)
        mel_spec = processed['mel_spectrogram']
        
        
        mel_spec = mel_spec.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        mel_spec = mel_spec.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(mel_spec)
            probabilities = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probabilities, dim=1)
        
        class_id = pred_class.item()
        confidence = probabilities[0, class_id].item()
        class_name = self.classes[class_id] if class_id < len(self.classes) else "unknown"
        
        is_alert = confidence >= self.threshold
        
        return PredictionResult(
            class_name=class_name,
            confidence=float(confidence),
            class_id=class_id,
            timestamp=time.time(),
            features=processed['features'],
            is_alert=is_alert
        )
    
    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {
                'total_predictions': self.total_predictions,
                'mean_time': 0,
                'std_time': 0,
                'fps': 0
            }
        
        mean_time = np.mean(self.inference_times)
        std_time = np.std(self.inference_times)
        
        return {
            'total_predictions': self.total_predictions,
            'mean_time': mean_time,
            'std_time': std_time,
            'fps': 1.0 / mean_time if mean_time > 0 else 0,
            'min_time': np.min(self.inference_times),
            'max_time': np.max(self.inference_times)
        }
    
    def export_to_onnx(self, onnx_path: str, input_shape: Tuple[int, ...] = (1, 3, 128, 128)):
        """Export model to ONNX format for deployment"""
        self.model.eval()
        dummy_input = torch.randn(*input_shape).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        logger.info(f"Model exported to ONNX: {onnx_path}")