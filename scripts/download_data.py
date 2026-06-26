#!/usr/bin/env python
"""Download and prepare sample data for environmental monitoring"""

import os
import sys
import argparse
from pathlib import Path
import requests
import zipfile
import tarfile
from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class DataDownloader:
    """Download and prepare environmental audio datasets"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_urban_sound(self):
        """Download UrbanSound8K dataset"""
        url = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"
        self._download_file(url, "UrbanSound8K.tar.gz")
        self._extract_tar("UrbanSound8K.tar.gz", self.data_dir)
        
    def download_esc50(self):
        """Download ESC-50 dataset"""
        url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
        self._download_file(url, "ESC-50-master.zip")
        self._extract_zip("ESC-50-master.zip", self.data_dir)
        
    def download_sample_data(self):
        """Generate synthetic sample data for testing"""
        logger.info("Generating sample audio data...")
        
        classes = ['chainsaw', 'gunshot', 'vehicle', 'fire', 'animal_distress', 'background']
        sr = 22050
        duration = 5.0
        
        for class_name in classes:
            class_dir = self.data_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            for i in range(10):  # Generate 10 samples per class
                # Generate synthetic audio
                audio = self._generate_synthetic_audio(class_name, sr, duration)
                
                # Save
                file_path = class_dir / f"{class_name}_{i:03d}.wav"
                sf.write(file_path, audio, sr)
        
        logger.info(f"Sample data generated in {self.data_dir}")
    
    def _generate_synthetic_audio(self, class_name: str, sr: int, duration: float) -> np.ndarray:
        """Generate synthetic audio for testing"""
        t = np.linspace(0, duration, int(sr * duration))
        
        if class_name == 'chainsaw':
            # Chainsaw-like sound
            freq = 200 + 50 * np.sin(2 * np.pi * 2 * t)
            audio = np.sin(2 * np.pi * freq * t)
            audio *= 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
            
        elif class_name == 'gunshot':
            # Gunshot-like sound
            audio = np.random.randn(len(t)) * np.exp(-t * 20)
            audio = np.clip(audio, -1, 1)
            
        elif class_name == 'vehicle':
            # Vehicle-like sound
            audio = np.sin(2 * np.pi * 100 * t) * 0.3
            audio += np.random.randn(len(t)) * 0.1
            
        elif class_name == 'fire':
            # Fire-like sound
            audio = np.random.randn(len(t)) * np.exp(-t * 2)
            audio *= 0.3
            
        elif class_name == 'animal_distress':
            # Animal distress-like sound
            audio = np.sin(2 * np.pi * 2000 * t) * np.exp(-10 * (t % 0.1 - 0.05)**2)
            audio += np.sin(2 * np.pi * 3000 * t) * np.exp(-10 * ((t + 0.05) % 0.1 - 0.05)**2)
            
        else:  # background
            # Background noise
            audio = np.random.randn(len(t)) * 0.1
            audio += 0.5 * np.sin(2 * np.pi * 50 * t) * 0.1
        
        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return audio
    
    def _download_file(self, url: str, filename: str):
        """Download a file with progress bar"""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            logger.info(f"{filename} already exists, skipping download")
            return
        
        logger.info(f"Downloading {filename}...")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    def _extract_zip(self, filename: str, extract_to: Path):
        """Extract zip file"""
        filepath = extract_to / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return
        
        logger.info(f"Extracting {filename}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Remove zip file
        filepath.unlink()
    
    def _extract_tar(self, filename: str, extract_to: Path):
        """Extract tar file"""
        filepath = extract_to / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return
        
        logger.info(f"Extracting {filename}...")
        with tarfile.open(filepath, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
        
        # Remove tar file
        filepath.unlink()


def main():
    parser = argparse.ArgumentParser(description='Download environmental audio data')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw',
        help='Directory to download data to'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['urban_sound', 'esc50', 'sample', 'all'],
        default='sample',
        help='Dataset to download'
    )
    
    args = parser.parse_args()
    
    downloader = DataDownloader(args.data_dir)
    
    if args.dataset == 'urban_sound':
        downloader.download_urban_sound()
    elif args.dataset == 'esc50':
        downloader.download_esc50()
    elif args.dataset == 'sample':
        downloader.download_sample_data()
    else:  # all
        downloader.download_urban_sound()
        downloader.download_esc50()
        downloader.download_sample_data()
    
    logger.info("Data download complete!")


if __name__ == "__main__":
    main()