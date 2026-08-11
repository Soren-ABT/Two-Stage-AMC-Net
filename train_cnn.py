#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用数据集：RML2018.01a，我的步骤：
  1. 多模态融合 (IQ + Amplitude + Phase = 4 通道)
  2. 两阶段训练 (Stage 1: SNR>=10; Stage 2: all SNR)两阶段训练对我的模型效果显著增强
  3. 显式正则化 (Label Smoothing = 0.1)
  保留增强操作、AdamW、CosineAnnealing、早停等。
"""

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import os

#配置
DATA_PATH = r".../GOLD_XYZ_OSC.0001_1024.hdf5"   #根据实际路径自己改
STAGE = 2
STAGE1_EPOCHS = 60
STAGE2_EPOCHS = 60
BATCH_SIZE = 256
LEARNING_RATE_STAGE1 = 0.001
LEARNING_RATE_STAGE2 = 0.0001
VAL_RATIO = 0.2
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_SMOOTHING = 0.1         # 只用这个显式正则化参数
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.1
GRAD_CLIP = 1.0

if STAGE == 1:
    SNR_TRAIN_RANGE = (10, 30)
    EPOCHS = STAGE1_EPOCHS
    LR = LEARNING_RATE_STAGE1
    MODEL_SAVE_PATH = "stage1_snr10_30.pth"
    PRETRAIN_PATH = None
else:
    SNR_TRAIN_RANGE = (-20, 30)
    EPOCHS = STAGE2_EPOCHS
    LR = LEARNING_RATE_STAGE2
    MODEL_SAVE_PATH = ".../Model stage/best_final_model.pth"   #第2阶段模型
    PRETRAIN_PATH = ".../Model stage/stage1_snr10_30.pth"   # 第1阶段模型

print(f"Stage: {STAGE}, SNR range: {SNR_TRAIN_RANGE}, LR: {LR}")

#多模态数据集处理
class RML2018MultimodalDataset(Dataset):
    def __init__(self, data_path, snr_range=None, augment=False):
        super().__init__()
        self.augment = augment
        with h5py.File(data_path, 'r') as f:
            X = f['X'][:]
            Y = f['Y'][:]
            Z = f['Z'][:]

        self.labels = np.argmax(Y, axis=1).astype(np.int64)
        self.snr = Z.flatten().astype(np.int32)

        # 保留原始 I/Q
        self.I = X[..., 0].astype(np.float32)
        self.Q = X[..., 1].astype(np.float32)

        if snr_range is not None:
            mask = (self.snr >= snr_range[0]) & (self.snr <= snr_range[1])
            self.I = self.I[mask]
            self.Q = self.Q[mask]
            self.labels = self.labels[mask]
            self.snr = self.snr[mask]

        #转torch张量
        self.I = torch.tensor(self.I, dtype=torch.float32)
        self.Q = torch.tensor(self.Q, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        I_raw = self.I[idx]   # (1024,)
        Q_raw = self.Q[idx]
        label = self.labels[idx]
        snr_val = self.snr[idx]

        if self.augment:
            #随机幅度缩放
            amp_scale = np.random.uniform(0.8, 1.2)
            I_raw = I_raw * amp_scale
            Q_raw = Q_raw * amp_scale
            #随机相位旋转
            theta = np.random.uniform(-np.pi/8, np.pi/8)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            I_rot = I_raw * cos_t - Q_raw * sin_t
            Q_rot = I_raw * sin_t + Q_raw * cos_t
            I_raw, Q_raw = I_rot, Q_rot
            #信号加噪
            I_raw += torch.randn_like(I_raw) * 0.02
            Q_raw += torch.randn_like(Q_raw) * 0.02

        #计算幅度与相位
        amplitude = torch.sqrt(I_raw**2 + Q_raw**2 + 1e-8)
        phase = torch.atan2(Q_raw, I_raw)

        #拼接为4通道(4, 1024)
        x = torch.stack([I_raw, Q_raw, amplitude, phase], dim=0)

        #逐样本标准化 (注意！！！已包含了幅度和相位)
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        x = (x - mean) / std

        return x, label, snr_val

#改进版ResNet(4 通道输入)
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, in_channels=4, num_classes=24, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout_rate=dropout_rate)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def ResNet18_1D_4ch(num_classes=24):
    return ResNet1D(BasicBlock1D, [2,2,2,2], in_channels=4, num_classes=num_classes)

#训练&评估函数
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for data, target, _ in tqdm(loader, desc="Train", leave=False):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    return total_loss/total, 100.*correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    snr_bins = np.arange(-20, 31, 2)
    snr_correct = {b:0 for b in snr_bins}
    snr_count = {b:0 for b in snr_bins}
    for data, target, snr in tqdm(loader, desc="Eval", leave=False):
        data, target, snr = data.to(device), target.to(device), snr.to(device)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        for b in snr_bins:
            mask = (snr == b)
            if mask.sum() > 0:
                snr_correct[b] += pred[mask].eq(target[mask]).sum().item()
                snr_count[b] += mask.sum().item()
    overall_acc = 100.*correct/total
    per_snr_acc = {b: (100.*snr_correct[b]/snr_count[b] if snr_count[b]>0 else 0.0) for b in snr_bins}
    return total_loss/total, overall_acc, per_snr_acc

# ==================== 主函数 ====================
def main():
    print("Loading data...")
    train_ds = RML2018MultimodalDataset(DATA_PATH, snr_range=SNR_TRAIN_RANGE, augment=True)
    val_ds   = RML2018MultimodalDataset(DATA_PATH, snr_range=SNR_TRAIN_RANGE, augment=False)

    labels_np = val_ds.labels.numpy()
    train_idx, val_idx = train_test_split(
        np.arange(len(train_ds)), test_size=VAL_RATIO, stratify=labels_np, random_state=RANDOM_SEED
    )
    train_set = Subset(train_ds, train_idx)
    val_set = Subset(val_ds, val_idx)
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set, BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = ResNet18_1D_4ch(num_classes=24).to(DEVICE)

    #if进入第二阶段，加载第一次的预训练权重
    if PRETRAIN_PATH is not None and STAGE == 2:
        print(f"Loading pretrained weights from {PRETRAIN_PATH}")
        model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=DEVICE))

    # 显式正则化：标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_snr_avg = 0.0
    patience_counter = 0
    early_patience = 20
    best_epoch = 0

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, snr_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        high_snr_bins = [s for s in snr_acc if s >= 0]
        if high_snr_bins:
            high_snr_avg = np.mean([snr_acc[s] for s in high_snr_bins])
            print(f"  >>> SNR >= 0 dB average accuracy: {high_snr_avg:.2f}% <<<")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc  # 新增
            best_snr_avg = high_snr_avg  # 新增
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  => Saved (val_loss {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_patience:
                print(f"Early stop at epoch {epoch}. Best val_loss {best_val_loss:.4f} at epoch {best_epoch}")
                break

        if epoch % 5 == 0:
            plt.figure(figsize=(10,5))
            snr_vals = sorted(snr_acc.keys())
            acc_vals = [snr_acc[s] for s in snr_vals]
            plt.plot(snr_vals, acc_vals, 'o-')
            plt.axhline(y=90, color='r', linestyle='--', label='90% target')
            plt.xlabel("SNR (dB)")
            plt.ylabel("Accuracy (%)")
            plt.title(f"Stage{STAGE} Epoch{epoch} – Per-SNR Accuracy")
            plt.legend(); plt.grid(True)
            plt.savefig(f"stage{STAGE}_epoch{epoch}.png", dpi=150)
            plt.close()

    # ==================== 生成 PDF 报告 ====================
    #找到最新的SNR曲线图
    img_files = [f for f in os.listdir('.') if f.startswith(f'stage{STAGE}') and f.endswith('.png')]
    img_file = sorted(img_files)[-1] if img_files else None

    if img_file:
        c = canvas.Canvas(f"AMC_Report_Stage{STAGE}.pdf", pagesize=A4)
        w, h = A4
        # 标题
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, h - 50, f"实验报告 —— Stage {STAGE}")
        # 关键指标
        c.setFont("Helvetica", 14)
        c.drawString(50, h - 100, f"最佳验证损失: {best_val_loss:.4f}")
        c.drawString(50, h - 125, f"最佳验证准确率: {best_val_acc:.2f}%")
        c.drawString(50, h - 150, f"最佳SNR>=0dB准确率: {best_snr_avg:.2f}%")
        c.drawString(50, h - 175, f"收敛轮次: Epoch {best_epoch}")
        # 嵌入曲线图
        c.drawImage(img_file, 50, h - 400, width=500, height=250)
        c.save()
        print(f"  => PDF report saved: AMC_Report_Stage{STAGE}.pdf")
    else:
        print("  => No SNR plot found, skip PDF generation.")

    print(f"Stage {STAGE} finished. Best val_loss: {best_val_loss:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()