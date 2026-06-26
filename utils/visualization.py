"""Visualization utilities for environmental monitoring"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa.display
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch


class Visualizer:
    """Visualization utilities for monitoring system"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_mel_spectrogram(self, mel_spec: np.ndarray, title: str = "Mel Spectrogram",
                             save: bool = True):
        """Plot Mel spectrogram"""
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            mel_spec, 
            x_axis='time', 
            y_axis='mel',
            sr=22050,
            hop_length=512
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        if save:
            self._save_plot('mel_spectrogram')
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                             save: bool = True):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues',
            cbar=True
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save:
            self._save_plot('confusion_matrix')
        plt.show()
    
    def plot_training_history(self, history: Dict, save: bool = True):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        if 'accuracy' in history:
            axes[0, 0].plot(history['accuracy'], label='Train')
            axes[0, 0].plot(history['val_accuracy'], label='Validation')
            axes[0, 0].set_title('Model Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
        
        # Loss
        if 'loss' in history:
            axes[0, 1].plot(history['loss'], label='Train')
            axes[0, 1].plot(history['val_loss'], label='Validation')
            axes[0, 1].set_title('Model Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
        
        # Learning rate
        if 'learning_rate' in history or 'learning_rates' in history:
            lrs = history.get('learning_rate', history.get('learning_rates', []))
            axes[1, 0].plot(lrs)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
        
        plt.tight_layout()
        if save:
            self._save_plot('training_history')
        plt.show()
    
    def plot_audio_waveform(self, audio: np.ndarray, sr: int = 22050,
                           title: str = "Audio Waveform", save: bool = True):
        """Plot audio waveform"""
        plt.figure(figsize=(12, 4))
        time = np.arange(len(audio)) / sr
        plt.plot(time, audio)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            self._save_plot('waveform')
        plt.show()
    
    def plot_alert_timeline(self, alerts: List[Dict], save: bool = True):
        """Plot alert timeline"""
        if not alerts:
            print("No alerts to plot")
            return
        
        df = pd.DataFrame(alerts)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Alerts by Event Type', 'Alerts by Severity'))
        
        # Alerts by event type
        if 'event_type' in df.columns:
            event_counts = df['event_type'].value_counts()
            fig.add_trace(
                go.Bar(x=event_counts.index, y=event_counts.values, 
                      name='Event Type', marker_color='lightblue'),
                row=1, col=1
            )
        
        # Alerts by severity
        if 'severity' in df.columns:
            severity_counts = df['severity'].value_counts()
            fig.add_trace(
                go.Bar(x=severity_counts.index, y=severity_counts.values,
                      name='Severity', marker_color='lightcoral'),
                row=2, col=1
            )
        
        fig.update_layout(height=600, title_text="Alert Timeline Analysis")
        
        if save:
            fig.write_html(self.output_dir / 'alert_timeline.html')
        fig.show()
    
    def plot_model_performance(self, metrics: Dict, save: bool = True):
        """Plot model performance metrics"""
        fig = go.Figure()
        
        # Bar chart for metrics
        metrics_to_plot = {k: v for k, v in metrics.items() 
                          if isinstance(v, (int, float)) and k != 'confusion_matrix'}
        
        fig.add_trace(go.Bar(
            x=list(metrics_to_plot.keys()),
            y=list(metrics_to_plot.values()),
            text=[f"{v:.3f}" for v in metrics_to_plot.values()],
            textposition='auto',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Model Performance Metrics",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400
        )
        
        if save:
            fig.write_html(self.output_dir / 'model_performance.html')
        fig.show()
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importance: List[float], save: bool = True):
        """Plot feature importance"""
        # Sort by importance
        sorted_idx = np.argsort(importance)
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_importance = [importance[i] for i in sorted_idx]
        
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_names, sorted_importance)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        if save:
            self._save_plot('feature_importance')
        plt.show()
    
    def _save_plot(self, name: str):
        """Save plot to output directory"""
        plt.savefig(self.output_dir / f'{name}.png', dpi=300, bbox_inches='tight')