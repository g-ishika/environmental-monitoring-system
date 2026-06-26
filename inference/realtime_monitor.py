"""Real-time monitoring for environmental audio"""

import torch
import numpy as np
import threading
import queue
import time
from pathlib import Path
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field
from collections import deque

from .predictor import Predictor, PredictionResult
from .alert_system import AlertSystem
from data.preprocessing import AudioPreprocessor
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring"""
    window_size: float = 5.0  # seconds
    overlap: float = 0.5  # 0-1
    sample_rate: int = 22050
    buffer_size: int = 10  # Number of windows to keep in buffer
    alert_cooldown: int = 60  # seconds
    threshold: float = 0.7  # confidence threshold for alerts
    
    def __post_init__(self):
        """Validate config values"""
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be between 0 and 1")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")


class RealTimeMonitor:
    """Real-time environmental audio monitor"""
    
    def __init__(
        self,
        predictor: Predictor,
        alert_system: AlertSystem,
        config: Dict,
        device: str = 'cuda'
    ):
        self.predictor = predictor
        self.alert_system = alert_system
        
        # Handle config properly - filter only valid keys for MonitoringConfig
        valid_keys = ['window_size', 'overlap', 'sample_rate', 'buffer_size', 'alert_cooldown', 'threshold']
        filtered_config = {k: v for k, v in config.items() if k in valid_keys}
        self.config = MonitoringConfig(**filtered_config)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.config.buffer_size)
        self.result_buffer = deque(maxlen=self.config.buffer_size)
        
        # Processing thread
        self.is_running = False
        self.process_thread = None
        
        # Queues for communication
        self.audio_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        
        # Statistics
        self.stats = {
            'processed_windows': 0,
            'alerts_triggered': 0,
            'total_processing_time': 0,
            'average_processing_time': 0,
            'start_time': None,
            'current_status': 'idle'
        }
        
        # Preprocessor
        self.preprocessor = AudioPreprocessor(
            sample_rate=self.config.sample_rate,
            duration=self.config.window_size
        )
        
        # Window calculation
        self.window_samples = int(self.config.sample_rate * self.config.window_size)
        self.hop_samples = int(self.window_samples * (1 - self.config.overlap))
        
        logger.info(f"RealTimeMonitor initialized with window_size={self.config.window_size}s, "
                   f"overlap={self.config.overlap}")
    
    def start(self, audio_source: Optional[Callable] = None):
        """Start real-time monitoring"""
        if self.is_running:
            logger.warning("Monitor already running")
            return
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # Start processing thread
        self.process_thread = threading.Thread(
            target=self._process_loop,
            args=(audio_source,),
            daemon=True
        )
        self.process_thread.start()
        
        logger.info("Real-time monitoring started")
        self.stats['current_status'] = 'running'
    
    def stop(self):
        """Stop real-time monitoring"""
        self.is_running = False
        if self.process_thread:
            self.process_thread.join(timeout=5)
        
        self.stats['current_status'] = 'stopped'
        logger.info("Real-time monitoring stopped")
    
    def feed_audio(self, audio_chunk: np.ndarray):
        """Feed audio chunk for processing"""
        if not self.is_running:
            logger.warning("Monitor not running, ignoring audio feed")
            return
        
        # Convert to float32 if needed
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Put in queue
        try:
            self.audio_queue.put_nowait(audio_chunk)
        except queue.Full:
            logger.warning("Audio queue full, dropping chunk")
    
    def _process_loop(self, audio_source: Optional[Callable] = None):
        """Main processing loop"""
        accumulated_audio = np.array([])
        
        while self.is_running:
            try:
                # Get audio chunk
                if audio_source is not None:
                    # Get from callback
                    chunk = audio_source(self.config.window_size)
                else:
                    # Get from queue
                    try:
                        chunk = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                
                # Accumulate audio
                accumulated_audio = np.concatenate([accumulated_audio, chunk])
                
                # Process windows
                while len(accumulated_audio) >= self.window_samples:
                    # Extract window
                    window = accumulated_audio[:self.window_samples]
                    accumulated_audio = accumulated_audio[self.hop_samples:]
                    
                    # Process window
                    self._process_window(window)
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(0.1)
    
    def _process_window(self, audio_window: np.ndarray):
        """Process a single audio window"""
        start_time = time.time()
        
        try:
            # Process audio
            processed = self.preprocessor.process_audio(
                audio_window, apply_noise_reduction=True
            )
            mel_spec = processed['mel_spectrogram']
            
            # Convert to image format
            mel_spec = mel_spec.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
            mel_spec = mel_spec.to(self.device)
            
            # Run prediction
            with torch.no_grad():
                outputs = self.predictor.model(mel_spec)
                probabilities = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probabilities, dim=1)
            
            class_id = pred_class.item()
            confidence = probabilities[0, class_id].item()
            class_name = self.predictor.classes[class_id] if class_id < len(self.predictor.classes) else "unknown"
            
            is_alert = confidence >= self.config.threshold
            
            # Create result
            result = PredictionResult(
                class_name=class_name,
                confidence=float(confidence),
                class_id=class_id,
                timestamp=time.time(),
                features=processed['features'],
                is_alert=is_alert,
                all_probs=probabilities.cpu().numpy()[0]
            )
            
            # Update stats
            self.stats['processed_windows'] += 1
            processing_time = time.time() - start_time
            self.stats['total_processing_time'] += processing_time
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['processed_windows']
            )
            
            # Store in buffer
            self.audio_buffer.append(audio_window)
            self.result_buffer.append(result)
            
            # Put in result queue
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                pass
            
            # Handle alert
            if is_alert:
                self.stats['alerts_triggered'] += 1
                self._handle_alert(result)
            
        except Exception as e:
            logger.error(f"Error processing window: {e}")
    
    def _handle_alert(self, result: PredictionResult):
        """Handle alert trigger"""
        alert = self.alert_system.create_alert(
            prediction_result={
                'class_name': result.class_name,
                'confidence': result.confidence,
                'features': result.features
            },
            location="Real-time Monitor"
        )
        
        self.alert_system.send_alert(alert)
        logger.info(f"ALERT: {result.class_name} ({result.confidence:.2%})")
    
    def get_latest_result(self) -> Optional[PredictionResult]:
        """Get the latest prediction result"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_performance_report(self) -> Dict:
        """Get performance report"""
        runtime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        
        return {
            'status': self.stats['current_status'],
            'runtime_seconds': runtime,
            'processed_windows': self.stats['processed_windows'],
            'alerts_triggered': self.stats['alerts_triggered'],
            'alerts_per_hour': (self.stats['alerts_triggered'] / (runtime / 3600)) if runtime > 0 else 0,
            'average_processing_time_ms': self.stats['average_processing_time'] * 1000,
            'fps': 1 / self.stats['average_processing_time'] if self.stats['average_processing_time'] > 0 else 0,
            'buffer_size': len(self.audio_buffer)
        }
    
    def get_buffer_stats(self) -> Dict:
        """Get statistics about the audio buffer"""
        if not self.audio_buffer:
            return {'size': 0, 'empty': True}
        
        return {
            'size': len(self.audio_buffer),
            'total_samples': sum(len(chunk) for chunk in self.audio_buffer),
            'duration_seconds': sum(len(chunk) for chunk in self.audio_buffer) / self.config.sample_rate,
            'empty': False
        }