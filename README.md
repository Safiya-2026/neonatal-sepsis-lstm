# LSTM Autoencoder for Neonatal Sepsis Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An unsupervised anomaly detection framework using an **LSTM Autoencoder** trained on neonatal physiological time-series data to detect early signs of sepsis.

---

## Overview

Neonatal sepsis is a life-threatening condition requiring early identification. This model learns the normal physiological patterns of neonates (heart rate, SpO₂, respiratory rate, temperature, blood pressure, blood glucose) and flags deviations as potential sepsis based on **reconstruction error**.

**Key idea:** Train only on healthy (non-sepsis) signals → high reconstruction error at inference = anomaly = sepsis alert.

---

## Architecture

```
Input (batch, seq_len, n_features)
        │
   ┌────▼────────────────┐
   │  Encoder (LSTM ×2)  │  → Latent vector (batch, latent_dim)
   └────────────────────┘
        │
   ┌────▼────────────────┐
   │  Decoder (LSTM ×2)  │  → Reconstruction (batch, seq_len, n_features)
   └─────────────────────┘
        │
   Reconstruction Error (MSE) → Anomaly Score → Threshold → Sepsis / Normal
```

---

## Input Features

| Feature | Unit |
|---|---|
| Heart Rate | bpm |
| Respiratory Rate | breaths/min |
| SpO₂ | % |
| Temperature | °C |
| Systolic BP | mmHg |
| Diastolic BP | mmHg |
| Mean Arterial Pressure | mmHg |
| Blood Glucose | mmol/L |

---

## Project Structure

```
neonatal-sepsis-lstm/
├── models/
│   ├── __init__.py
│   └── lstm_autoencoder.py     # Encoder, Decoder, LSTMAutoencoder
├── data/
│   └── preprocessing.py        # Cleaning, windowing, normalization
├── results/                    # Saved model, plots
├── tests/
│   └── test_model.py
├── train.py                    # Training script
├── evaluate.py                 # Evaluation + threshold selection
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/neonatal-sepsis-lstm.git
cd neonatal-sepsis-lstm
pip install -r requirements.txt
```

---

## Quick Start

### 1. Generate synthetic data (for testing)
```bash
python -c "from data.preprocessing import generate_synthetic_data; generate_synthetic_data()"
```

### 2. Train the model
```bash
python train.py \
  --data_path data/normal_signals.npy \
  --input_dim 8 \
  --seq_len 24 \
  --hidden_dim 64 \
  --latent_dim 32 \
  --epochs 100 \
  --batch_size 64
```

### 3. Evaluate and set threshold
```bash
python evaluate.py \
  --normal_data data/test_normal.npy \
  --sepsis_data data/test_sepsis.npy \
  --model_path results/best_model.pt \
  --threshold_percentile 95
```

---

## Code Availability

The complete source code for the LSTM Autoencoder model, preprocessing pipeline, training, and evaluation scripts is publicly available in this repository. A permanently archived version is deposited on Zenodo:

> **DOI:** `https://doi.org/[YOUR-ZENODO-DOI]`  ← *Replace after Zenodo deposit*

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{yourname2024neonatal,
  author    = {[Your Name]},
  title     = {LSTM Autoencoder for Neonatal Sepsis Detection},
  year      = {2024},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/<your-username>/neonatal-sepsis-lstm}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
