"""Audio utilities for environmental monitoring"""

import torch
import torchaudio
import librosa
import numpy as np
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')


class AudioUtils:
    """Utility functions for audio processing"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def load_audio(self, path: str, normalize: bool = True) -> np.ndarray:
        """Load audio file"""
        audio, sr = librosa.load(path, sr=self.sample_rate)
        
        if normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio
    
    def save_audio(self, audio: np.ndarray, path: str):
        """Save audio file"""
        sf.write(path, audio, self.sample_rate)
    
    def get_duration(self, audio: np.ndarray) -> float:
        """Get audio duration in seconds"""
        return len(audio) / self.sample_rate
    
    def trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from beginning and end"""
        non_silent = np.where(np.abs(audio) > threshold)[0]
        if len(non_silent) == 0:
            return audio
        
        start = non_silent[0]
        end = non_silent[-1]
        return audio[start:end+1]
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc
        )
        return mfccs
    
    def extract_spectrogram(self, audio: np.ndarray, n_fft: int = 2048,
                           hop_length: int = 512) -> np.ndarray:
        """Extract spectrogram"""
        spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        return np.abs(spectrogram)
    
    def extract_mel_spectrogram(self, audio: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """Extract Mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_features(self, audio: np.ndarray) -> Dict:
        """Extract comprehensive audio features"""
        features = {}
        
        # MFCCs
        mfccs = self.extract_mfcc(audio)
        features['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
        features['mfcc_std'] = np.std(mfccs, axis=1).tolist()
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate
        )
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate
        )
        
        features['spectral_centroid'] = float(np.mean(spectral_centroids))
        features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
        features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
        features['spectral_contrast'] = np.mean(spectral_contrast, axis=1).tolist()
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr'] = float(np.mean(zcr))
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        features['rms'] = float(np.mean(rms))
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
        features['tempo'] = float(tempo)
        
        # Harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio)
        features['harmonic_energy'] = float(np.mean(np.abs(harmonic)))
        features['percussive_energy'] = float(np.mean(np.abs(percussive)))
        
        return features
    
    def calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS value"""
        return float(np.sqrt(np.mean(audio ** 2)))
    
    def calculate_peak(self, audio: np.ndarray) -> float:
        """Calculate peak value"""
        return float(np.max(np.abs(audio)))
    
    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
        return float(snr)
    
    def mix_audio(self, audio1: np.ndarray, audio2: np.ndarray, 
                  weight1: float = 0.5) -> np.ndarray:
        """Mix two audio signals"""
        # Ensure same length
        max_len = max(len(audio1), len(audio2))
        if len(audio1) < max_len:
            audio1 = np.pad(audio1, (0, max_len - len(audio1)))
        if len(audio2) < max_len:
            audio2 = np.pad(audio2, (0, max_len - len(audio2)))
        
        # Mix
        mixed = weight1 * audio1 + (1 - weight1) * audio2
        return mixed
    
    def apply_effects(self, audio: np.ndarray, effects: Dict) -> np.ndarray:
        """Apply audio effects"""
        result = audio.copy()
        
        if effects.get('reverb', False):
            # Simple reverb (convolution with exponential decay)
            impulse = np.exp(-np.arange(0, 0.3, 1/self.sample_rate))
            result = np.convolve(result, impulse, mode='same')
        
        if effects.get('chorus', False):
            # Simple chorus (delay with modulation)
            delay = int(0.03 * self.sample_rate)
            modulated = np.roll(result, delay)
            result = result + 0.3 * modulated
            result = result / np.max(np.abs(result))
        
        if effects.get('distortion', 0) > 0:
            # Simple distortion
            gain = effects.get('distortion')
            result = np.tanh(gain * result)
        
        return result