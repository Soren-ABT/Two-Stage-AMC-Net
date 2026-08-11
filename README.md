# Two-Stage-AMC-Net

**Two-Stage-AMC-Net**是一个用于自动调制识别（Automatic Modulation Classification, AMC）的深度学习模型。该模型基于 1D ResNet-18 骨架，采用 **4 通道多模态输入（I/Q + 幅度 + 相位）**，结合**两阶段课程训练策略**与**标签平滑正则化**，在 RadioML 2018.01A 数据集上实现了 SNR ≥ 0 dB 条件下 **92%+ 的分类准确率**。

---

## 环境要求

### 硬件

| 组件 | 最低要求 | 推荐 |
|------|---------|------|
| GPU | NVIDIA 显存 ≥ 6 GB | RTX 3060 及以上 |
| 内存 | 16 GB | 32 GB |
| 硬盘 | 20 GB 空闲 | SSD |


### 软件

| 软件 | 版本 |
|------|------|
| Python | 3.9 ~ 3.11 |
| PyTorch | ≥ 2.0 |
| CUDA | 11.8 / 12.1 |

---

## 数据集说明

本项目使用 **RadioML 2018.01A** 数据集（`GOLD_XYZ_OSC.0001_1024.hdf5`）。

### 数据维度

| 数组 | 形状 | 含义 |
|------|------|------|
| `X` | (2555904, 1024, 2) | I/Q 复数信号，`[...,0]` 为 I 路，`[...,1]` 为 Q 路 |
| `Y` | (2555904, 24) | One‑Hot 编码的调制类型标签 |
| `Z` | (2555904, 1) | 信噪比（SNR），范围 -20 ~ 30 dB，步长 2 dB |

### 24 种调制类型

| 类别 | 名称 | 类别 | 名称 |
|------|------|------|------|
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

## 快速开始

### 1. 获取代码与数据

```bash
git clone https://github.com/Soren-ABT/Two-Stage-AMC-Net.git
cd Two-Stage-AMC-Net
```
修改 `train_cnn.py` 中的 `DATA_PATH` 变量。

### 2. 第一阶段训练（高 SNR 预训练）

确认 `train_cnn.py` 中 `STAGE = 1`，然后运行：

```bash
python train_cnn.py
```

训练结束后生成 `stage1_snr10_30.pth` 和 `AMC_Report_Stage1.pdf`。

### 3. 第二阶段训练（全 SNR 微调）

将 `STAGE` 改为 `2`，再次运行：

```bash
python train_cnn.py
```

训练结束后生成 `best_final_model.pth` 和 `AMC_Report_Stage2.pdf`。

---

## 项目结构

```
Two-Stage-AMC-Net/
├── train_cnn.py                  # 主训练脚本
├── README.md                    
├── LICENSE             
└── Model stage                   # 两阶段训练权重及其我的运行图片
    ├── best_final_model.pth
    └── stage1_snr10_30.pth
```

---

## 模型架构

```
输入: (Batch, 4, 1024)   ← I, Q, Amplitude, Phase 四通道
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
输出: (Batch, 24)   ← 24 类 logits
```

### BasicBlock 结构

```
Input ──┬── Conv1d + BN + ReLU ── Conv1d + BN ── Dropout ──┬── + ── ReLU ── Output
        │                                                  │
        └────────  Shortcut (1×1 Conv if dim mismatch) ────┘
```

---

## 超参数配置

| 参数 | 第一阶段 | 第二阶段 | 说明 |
|------|---------|---------|------|
| `SNR_TRAIN_RANGE` | (10, 30) | (-20, 30) | 训练用 SNR 范围 |
| `EPOCHS` | 60 | 60 | 最大训练轮数 |
| `BATCH_SIZE` | 256 | 256 | 批大小 |
| `LEARNING_RATE` | 0.001 | 0.0001 | 学习率 |
| `LABEL_SMOOTHING` | 0.1 | 0.1 | 标签平滑系数 |
| `WEIGHT_DECAY` | 0.01 | 0.01 | AdamW 权重衰减 |
| `DROPOUT_RATE` | 0.1 | 0.1 | 残差块内 Dropout 比例 |
| `GRAD_CLIP` | 1.0 | 1.0 | 梯度裁剪阈值 |
| `VAL_RATIO` | 0.2 | 0.2 | 验证集比例 |
| `EARLY_PATIENCE` | 15 | 15 | 早停耐心值 |

---
