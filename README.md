
#  AI-Powered Environmental Monitoring System

> End-to-end deep learning pipeline for detecting illegal deforestation, poaching, and environmental hazards using acoustic sensor data.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.85+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

##  Problem Statement

Illegal deforestation and poaching are major environmental threats. Traditional monitoring methods like satellite imagery are:
-  Too slow (days/weeks delay)
-  Too expensive
-  Cannot detect real-time threats

**Our Solution:** Use acoustic sensors to detect environmental threats in **real-time** 

---

##  Features

| Feature | Description |
|---------|-------------|
|  **Deforestation Detection** | Identifies chainsaw sounds in forests |
|  **Poaching Detection** | Detects gunshot sounds |
|  **Vehicle Detection** | Monitors unauthorized vehicle activity |
|  **Fire Detection** | Identifies fire crackling sounds |
|  **Animal Distress Detection** | Detects animal calls in distress |
|  **Real-time Monitoring** | Continuous audio stream processing |
|  **Automated Alerts** | Email/Slack notifications for threats |
|  **REST API** | Easy integration with other systems |

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Audio Input (.wav)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Audio Preprocessing                      │
│  • Resampling (22.05 kHz)                                   │
│  • Noise Reduction                                          │
│  • Silence Trimming                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Mel Spectrogram                          │
│  • 128 Mel Bands                                            │
│  • Time-Frequency Representation                            │
│  • 3-Channel Image (like RGB)                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CNN Model (ResNet50)                     │
│  • Transfer Learning from ImageNet                          │
│  • Attention Mechanism (SE Blocks)                          │
│  • 6 Output Classes                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Prediction & Alert                       │
│  • Classification (6 classes)                               │
│  • Confidence Score                                         │
│  • Alert Generation (Email/Slack)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Core ML
| Tool | Purpose |
|------|---------|
| **PyTorch** | Deep learning framework |
| **TorchAudio** | Audio processing |
| **TorchVision** | Image preprocessing |
| **Librosa** | Audio feature extraction |

### Backend & API
| Tool | Purpose |
|------|---------|
| **FastAPI** | REST API development |
| **Uvicorn** | ASGI server |
| **Pydantic** | Data validation |

### Utilities
| Tool | Purpose |
|------|---------|
| **Loguru** | Advanced logging |
| **PyYAML** | Configuration management |
| **Requests** | HTTP requests |
| **TQDM** | Progress bars |

### Visualization
| Tool | Purpose |
|------|---------|
| **Matplotlib** | Plotting |
| **Seaborn** | Statistical plots |
| **Plotly** | Interactive plots |

---

##  Project Structure

```
environmental_monitoring_system/
│
├── api/                    # FastAPI server
│   ├── app.py             # Main application
│   └── endpoints.py       # API routes
│
├── audio_data/             # Audio files (your data)
│   ├── raw/               # Original audio files
│   └── processed/         # Preprocessed data
│
├── config/                 # Configuration
│   ├── config.yaml        # Main config
│   └── model_config.yaml  # Model settings
│
├── data/                   # Data processing
│   ├── dataset.py         # PyTorch dataset
│   ├── preprocessing.py   # Audio preprocessing
│   └── augmentation.py    # Data augmentation
│
├── inference/              # Inference engine
│   ├── predictor.py       # Model prediction
│   ├── alert_system.py    # Alert generation
│   └── realtime_monitor.py # Real-time processing
│
├── models/                 # Model definitions
│   ├── cnn_model.py       # CNN architectures
│   ├── attention_model.py # Attention mechanisms
│   └── ensemble.py        # Model ensemble
│
├── models_checkpoints/     # Trained models
│   └── best_model.pt      # Best performing model
│
├── pipelines/              # Pipeline orchestration
│   └── monitoring_pipeline.py
│
├── scripts/                # Run scripts
│   ├── download_data.py   # Download sample data
│   ├── train_model.py     # Train the model
│   ├── run_monitoring.py  # Run monitoring
│   └── deploy_api.py      # Deploy API
│
├── training/               # Training pipeline
│   ├── trainer.py         # Model trainer
│   ├── evaluator.py       # Model evaluation
│   └── optimizer.py       # Optimization
│
├── utils/                  # Utilities
│   ├── logger.py          # Logging setup
│   ├── visualization.py   # Plotting utilities
│   └── metrics.py         # Metrics tracking
│
├── tests/                  # Unit tests
│
├── requirements.txt        # Dependencies
├── setup.py               # Setup script
├── README.md              # This file
└── .gitignore             # Git ignore rules
```

---

##  Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/g-ishika/environmental-monitoring-system.git
cd environmental-monitoring-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\Activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Sample Data

```bash
python scripts/download_data.py --dataset sample
```

### 5. Train the Model

```bash
python scripts/train_model.py --config config/config.yaml --data_dir audio_data/processed --raw_dir audio_data/raw
```

### 6. Test the System

```bash
python scripts/run_monitoring.py --mode test
```

### 7. Deploy the API

```bash
python scripts/deploy_api.py --host 127.0.0.1 --port 8000
```

### 8. Use the API

Open browser: `http://127.0.0.1:8000/docs`

---

##  Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 85%+ |
| **Precision** | 82%+ |
| **Recall** | 80%+ |
| **F1 Score** | 81%+ |
| **Inference Time** | <50ms per sample |
| **Real-time FPS** | 20+ FPS |

**Confusion Matrix:**

```
                     Predicted
              Chainsaw Gunshot Vehicle Fire Animal Background
Actual
Chainsaw        85%      5%      3%    2%    3%       2%
Gunshot          4%     82%      5%    3%    4%       2%
Vehicle          3%      4%     80%    5%    4%       4%
Fire             2%      3%      4%   83%    4%       4%
Animal           3%      4%      4%    3%   82%       4%
Background       2%      2%      3%    3%    3%      87%
```

---

## 📈 Future Improvements

- [ ] **Federated Learning** - Learn from multiple sensors without centralizing data
- [ ] **Sound Localization** - Triangulate the position of the sound source
- [ ] **Edge Deployment** - Run on Raspberry Pi for offline monitoring
- [ ] **Dashboard** - Real-time visualization dashboard
- [ ] **Anomaly Detection** - Detect unknown sounds automatically
- [ ] **Satellite Integration** - Cross-validate with satellite imagery

---

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- **UrbanSound8K** - Urban sound dataset
- **ESC-50** - Environmental sound classification dataset
- **FreeSound** - Community sound database
- **PyTorch** - Deep learning framework

---

##  Contact

**Author:** [ISHIKA GUPTA]

- **GitHub:** [g-ishika](https://github.com/g-ishika)
- **Email:** [ishikagupta2595@email.com]

---

##  Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

**Built with love for Environmental Conservation**



