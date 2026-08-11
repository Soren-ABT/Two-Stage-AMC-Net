# Two-Stage-AMC-Net

**Two-Stage-AMC-Net** is a deep learning model for Automatic Modulation Classification (AMC). It is built upon a 1D ResNet-18 backbone and adopts **4-channel multi-modal inputs (I/Q + Amplitude + Phase)**, combined with a **two-stage curriculum training strategy** and **label smoothing regularization**. On the RadioML 2018.01A dataset, it achieves **92%+ classification accuracy** under SNR ≥ 0 dB.

---

**Languages:**
[**English**](./README.md) / [**中文**](./README.zh.md)

---
## Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA VRAM ≥ 6 GB | RTX 3060 or higher |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB free | SSD |

### Software

| Software | Version |
|----------|---------|
| Python | 3.9 ~ 3.11 |
| PyTorch | ≥ 2.0 |
| CUDA | 11.8 / 12.1 |

---

## Dataset Description

This project uses the **RadioML 2018.01A** dataset (`GOLD_XYZ_OSC.0001_1024.hdf5`).

### Data Dimensions

| Array | Shape | Description |
|-------|-------|-------------|
| `X` | (2555904, 1024, 2) | I/Q complex signals, `[...,0]` = I, `[...,1]` = Q |
| `Y` | (2555904, 24) | One‑hot encoded modulation labels |
| `Z` | (2555904, 1) | SNR (dB), range –20 ~ 30 dB, step 2 dB |

### 24 Modulation Types

| Class | Name | Class | Name |
|-------|------|-------|------|
| OOK | On‑Off Keying | 32APSK | 32‑Amplitude‑Phase Shift Keying |
| 4ASK | 4‑Amplitude Shift Keying | 64APSK | 64‑APSK |
| 8ASK | 8‑ASK | 128APSK | 128‑APSK |
| BPSK | Binary Phase Shift Keying | 16QAM | 16‑Quadrature Amplitude Mod. |
| QPSK | Quadrature PSK | 32QAM | 32‑QAM |
| 8PSK | 8‑PSK | 64QAM | 64‑QAM |
| 16PSK | 16‑PSK | 128QAM | 128‑QAM |
| 32PSK | 32‑PSK | 256QAM | 256‑QAM |
| 16APSK | 16‑APSK | GMSK | Gaussian Minimum Shift Keying |
| OQPSK | Offset QPSK | FM | Frequency Modulation |
| AM‑SSB‑WC | AM Single‑Sideband (With Carrier) | AM‑SSB‑SC | AM Single‑Sideband (Suppressed Carr.) |
| AM‑DSB‑WC | AM Double‑Sideband (With Carrier) | AM‑DSB‑SC | AM Double‑Sideband (Suppressed Carr.) |

---

## Quick Start

### 1. Get Code and Data

```bash
git clone https://github.com/Soren-ABT/Two-Stage-AMC-Net.git
cd Two-Stage-AMC-Net
```

Modify the `DATA_PATH` variable in `train_cnn.py` to point to your dataset location.

### 2. Stage 1 Training (High‑SNR Pre‑training)

Set `STAGE = 1` in `train_cnn.py`, then run:

```bash
python train_cnn.py
```

After training, `stage1_snr10_30.pth` and `AMC_Report_Stage1.pdf` will be generated.

### 3. Stage 2 Training (Full‑SNR Fine‑tuning)

Change `STAGE` to `2` and run again:

```bash
python train_cnn.py
```

The final model `best_final_model.pth` and `AMC_Report_Stage2.pdf` will be produced.

---

## Project Structure

```
Two-Stage-AMC-Net/
├── train_cnn.py                  # Main training script
├── README.md                    
├── LICENSE             
└── Model stage                   # Two‑stage weights and sample figures
    ├── best_final_model.pth
    └── stage1_snr10_30.pth
```

---

## Model Architecture

```
Input: (Batch, 4, 1024)   ← I, Q, Amplitude, Phase (4 channels)
       │
  ┌────▼─────────────────────────────────────┐
  │  Conv1d(4→64, k=7, s=2) + BN + ReLU      │
  │  MaxPool1d(k=3, s=2)                     │
  │                    ↓                     │
  │  Layer1: [BasicBlock(64→64)]   × 2       │
  │  Layer2: [BasicBlock(64→128)]  × 2       │
  │  Layer3: [BasicBlock(128→256)] × 2       │
  │  Layer4: [BasicBlock(256→512)] × 2       │
  │                    ↓                     │
  │  AdaptiveAvgPool1d(1)  → (B, 512, 1)     │
  │  Flatten               → (B, 512)        │
  │  Linear(512, 24)       → (B, 24)         │
  └──────────────────────────────────────────┘
Output: (Batch, 24)   ← 24‑class logits
```

### BasicBlock Structure

```
Input ──┬── Conv1d + BN + ReLU ── Conv1d + BN ── Dropout ──┬── + ── ReLU ── Output
        │                                                  │
        └────────  Shortcut (1×1 Conv if dim mismatch) ────┘
```

---

## Hyperparameter Configuration

| Parameter | Stage 1 | Stage 2 | Description |
|-----------|---------|---------|-------------|
| `SNR_TRAIN_RANGE` | (10, 30) | (-20, 30) | SNR range for training |
| `EPOCHS` | 60 | 60 | Maximum epochs |
| `BATCH_SIZE` | 256 | 256 | Batch size |
| `LEARNING_RATE` | 0.001 | 0.0001 | Learning rate |
| `LABEL_SMOOTHING` | 0.1 | 0.1 | Label smoothing factor |
| `WEIGHT_DECAY` | 0.01 | 0.01 | AdamW weight decay |
| `DROPOUT_RATE` | 0.1 | 0.1 | Dropout ratio inside residual blocks |
| `GRAD_CLIP` | 1.0 | 1.0 | Gradient clipping threshold |
| `VAL_RATIO` | 0.2 | 0.2 | Validation split ratio |
| `EARLY_PATIENCE` | 15 | 15 | Early stopping patience |
