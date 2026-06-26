"""Audio preprocessing pipeline using PyTorch"""

import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AudioPreprocessor:
    """Audio preprocessing pipeline for environmental monitoring using PyTorch"""
    
    sample_rate: int = 22050
    duration: float = 5.0
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    device: str = "cpu"
    
    def __post_init__(self):
        self.n_samples = int(self.sample_rate * self.duration)
    
    def load_audio(self, file_path: str) -> torch.Tensor:
        """Load and resample audio file"""
        try:
            # Use torchaudio for efficient loading
            waveform, sr = torchaudio.load(file_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Pad or truncate to fixed length
            if waveform.shape[1] < self.n_samples:
                pad_length = self.n_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            else:
                waveform = waveform[:, :self.n_samples]
            
            return waveform.squeeze(0)  # Remove channel dimension
            
        except Exception as e:
            raise RuntimeError(f"Failed to load audio {file_path}: {e}")
    
    def apply_noise_reduction(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply spectral noise reduction"""
        # FIX: Use detach() before converting to numpy
        audio_np = audio.detach().numpy()  # ← FIXED THIS LINE
        
        # Compute noise profile from first 0.5 seconds
        noise_samples = int(0.5 * self.sample_rate)
        if len(audio_np) > noise_samples:
            noise_profile = np.mean(np.abs(audio_np[:noise_samples]))
        else:
            noise_profile = np.mean(np.abs(audio_np))
        
        # Compute STFT
        stft = librosa.stft(audio_np, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Spectral subtraction
        magnitude_denoised = np.maximum(magnitude - noise_profile * 0.1, 0)
        
        # Reconstruct
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(stft_denoised, hop_length=self.hop_length)
        
        # Ensure same length
        if len(audio_denoised) < len(audio_np):
            audio_denoised = np.pad(audio_denoised, (0, len(audio_np) - len(audio_denoised)))
        else:
            audio_denoised = audio_denoised[:len(audio_np)]
        
        # Convert back to tensor
        return torch.from_numpy(audio_denoised).float()
    
    def extract_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract Mel spectrogram using torchaudio"""
        # Ensure audio is 1D or 2D
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Create Mel spectrogram transform
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )
        
        # Compute mel spectrogram
        mel_spec = mel_transform(audio)
        
        # Convert to log scale (dB)
        mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        # Normalize to [0, 1]
        mel_min = mel_spec_db.min()
        mel_max = mel_spec_db.max()
        if mel_max > mel_min:
            mel_spec_db = (mel_spec_db - mel_min) / (mel_max - mel_min + 1e-8)
        
        return mel_spec_db.squeeze(0)
    
    def extract_features(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract comprehensive features"""
        features = {}
        audio_np = audio.detach().numpy()  # ← FIXED THIS TOO
        
        # MFCCs using torchaudio
        try:
            if audio.dim() == 1:
                audio_2d = audio.unsqueeze(0)
            else:
                audio_2d = audio
            
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=13,
                log_mels=True
            )
            mfccs = mfcc_transform(audio_2d)
            features['mfcc_mean'] = torch.mean(mfccs, dim=2).squeeze(0)
            features['mfcc_std'] = torch.std(mfccs, dim=2).squeeze(0)
        except:
            # Fallback to librosa
            mfccs = librosa.feature.mfcc(y=audio_np, sr=self.sample_rate, n_mfcc=13)
            features['mfcc_mean'] = torch.tensor(np.mean(mfccs, axis=1))
            features['mfcc_std'] = torch.tensor(np.std(mfccs, axis=1))
        
        # Spectral features using librosa
        try:
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_np, sr=self.sample_rate
            )
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_np, sr=self.sample_rate
            )
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_np, sr=self.sample_rate
            )
            
            features['spectral_centroid'] = torch.tensor(np.mean(spectral_centroids))
            features['spectral_rolloff'] = torch.tensor(np.mean(spectral_rolloff))
            features['spectral_bandwidth'] = torch.tensor(np.mean(spectral_bandwidth))
        except:
            features['spectral_centroid'] = torch.tensor(0.0)
            features['spectral_rolloff'] = torch.tensor(0.0)
            features['spectral_bandwidth'] = torch.tensor(0.0)
        
        # Zero crossing rate
        try:
            zcr = librosa.feature.zero_crossing_rate(audio_np)
            features['zcr'] = torch.tensor(np.mean(zcr))
        except:
            features['zcr'] = torch.tensor(0.0)
        
        # RMS energy
        try:
            rms = librosa.feature.rms(y=audio_np)
            features['rms'] = torch.tensor(np.mean(rms))
        except:
            features['rms'] = torch.tensor(0.0)
        
        return features
    
    def process_audio(self, audio_or_path, apply_noise_reduction: bool = True) -> Dict:
        """Complete audio processing pipeline"""
        # Load audio
        if isinstance(audio_or_path, str):
            audio = self.load_audio(audio_or_path)
        else:
            audio = audio_or_path
            if len(audio) > self.n_samples:
                audio = audio[:self.n_samples]
            elif len(audio) < self.n_samples:
                pad_length = self.n_samples - len(audio)
                audio = torch.nn.functional.pad(audio, (0, pad_length))
        
        # Apply noise reduction
        if apply_noise_reduction:
            audio = self.apply_noise_reduction(audio)
        
        # Extract Mel spectrogram
        mel_spec = self.extract_mel_spectrogram(audio)
        
        # Extract additional features
        features = self.extract_features(audio)
        
        return {
            'mel_spectrogram': mel_spec,
            'features': features,
            'audio': audio
        }
    
    def mel_to_image(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to 3-channel image for CNN"""
        # Add channel dimension and repeat for RGB
        mel_spec = mel_spec.unsqueeze(0)  # (1, H, W)
        mel_spec = mel_spec.repeat(3, 1, 1)  # (3, H, W)
        return mel_spec