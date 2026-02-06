# -*- coding: utf-8 -*-
"""
LSTM+Attention 模型训练
AdamW + CosineAnnealingLR + 加权 Huber Loss + Early Stopping
"""

import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from deep_learning.dl_config import (
    DL_SEQUENCES_DIR, DL_MODELS_DIR, TRAIN_CONFIG, MODEL_CONFIG
)
from deep_learning.data.dataset import create_dataloaders
from deep_learning.models.lstm_attention import (
    build_model, WeightedHuberLoss
)


class EarlyStopping:
    """Early Stopping 监控"""

    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def train_one_epoch(model, loader, criterion, optimizer, device, clip_norm, noise_std=0.0):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 训练时添加高斯噪声（数据增强）
        if noise_std > 0:
            noise = torch.randn_like(X_batch) * noise_std
            X_batch = X_batch + noise

        optimizer.zero_grad()
        predictions, _ = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()

        # Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions, _ = model(X_batch)
            loss = criterion(predictions, y_batch)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train_model():
    """完整训练流程"""
    print("=" * 60)
    print("  LSTM+Attention 模型训练")
    print("=" * 60)

    # 设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\n🖥️ 使用设备: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n🖥️ 使用设备: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"\n🖥️ 使用设备: CPU")

    # 创建 DataLoader
    print("\n📦 加载数据...")
    loaders = create_dataloaders(
        DL_SEQUENCES_DIR,
        batch_size=TRAIN_CONFIG['batch_size'],
    )

    train_loader = loaders['train']
    val_loader = loaders['val']

    # 获取输入维度
    sample_X, sample_y = next(iter(train_loader))
    input_size = sample_X.shape[2]
    print(f"   输入维度: {input_size}")
    print(f"   序列长度: {sample_X.shape[1]}")
    print(f"   输出维度: {sample_y.shape[1]}")
    print(f"   训练集: {len(train_loader.dataset)} 样本")
    print(f"   验证集: {len(val_loader.dataset)} 样本")

    # 构建模型
    print("\n🏗️ 构建模型...")
    MODEL_CONFIG['input_size'] = input_size
    model = build_model(input_size, MODEL_CONFIG)
    model = model.to(device)

    # 损失函数
    criterion = WeightedHuberLoss(
        weights=TRAIN_CONFIG['horizon_weights'],
        delta=TRAIN_CONFIG['huber_delta'],
    ).to(device)

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
    )

    # 学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=TRAIN_CONFIG['max_epochs'],
        eta_min=1e-6,
    )

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=TRAIN_CONFIG['early_stopping_patience']
    )

    # 训练循环
    print(f"\n🚀 开始训练 (最多 {TRAIN_CONFIG['max_epochs']} epochs)...")
    print(f"   Batch Size: {TRAIN_CONFIG['batch_size']}")
    print(f"   Learning Rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"   Weight Decay: {TRAIN_CONFIG['weight_decay']}")
    print(f"   Gradient Clip: {TRAIN_CONFIG['gradient_clip_max_norm']}")
    noise_std = TRAIN_CONFIG.get('noise_std', 0.0)
    print(f"   Early Stopping Patience: {TRAIN_CONFIG['early_stopping_patience']}")
    if noise_std > 0:
        print(f"   Noise Std: {noise_std}")

    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
    }

    start_time = time.time()

    for epoch in range(1, TRAIN_CONFIG['max_epochs'] + 1):
        epoch_start = time.time()

        # 训练
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            TRAIN_CONFIG['gradient_clip_max_norm'],
            noise_std=noise_std,
        )

        # 验证
        val_loss = validate(model, val_loader, criterion, device)

        # 学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 记录
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'model_config': MODEL_CONFIG,
                'input_size': input_size,
            }
            torch.save(checkpoint, DL_MODELS_DIR / "best_model.pt")
            marker = " ★ best"
        else:
            marker = ""

        # 打印进度
        if epoch % 5 == 0 or epoch == 1 or marker:
            print(f"   Epoch {epoch:3d}/{TRAIN_CONFIG['max_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s{marker}")

        # Early Stopping
        if early_stopping(val_loss):
            print(f"\n   ⏹️ Early Stopping at epoch {epoch} "
                  f"(patience={TRAIN_CONFIG['early_stopping_patience']})")
            break

    total_time = time.time() - start_time
    print(f"\n✅ 训练完成!")
    print(f"   总耗时: {total_time/60:.1f} 分钟")
    print(f"   最佳验证 Loss: {best_val_loss:.4f} (epoch {checkpoint['epoch']})")

    # 保存训练曲线
    history_file = DL_MODELS_DIR / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"   训练曲线: {history_file}")

    # 保存最终模型（非最佳）
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'model_config': MODEL_CONFIG,
        'input_size': input_size,
    }
    torch.save(final_checkpoint, DL_MODELS_DIR / "final_model.pt")

    return model, training_history


def main():
    train_model()


if __name__ == "__main__":
    main()
