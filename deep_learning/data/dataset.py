# -*- coding: utf-8 -*-
"""
PyTorch Dataset 类
加载 .npy 序列用于模型训练
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class AQISequenceDataset(Dataset):
    """AQI 时序预测数据集"""

    def __init__(self, X_path, y_path):
        """
        Args:
            X_path: 输入序列 .npy 文件路径
            y_path: 目标值 .npy 文件路径
        """
        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)

        assert len(self.X) == len(self.y), \
            f"X ({len(self.X)}) 和 y ({len(self.y)}) 长度不匹配"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.y[idx]),
        )

    @property
    def n_features(self):
        return self.X.shape[2]

    @property
    def seq_len(self):
        return self.X.shape[1]

    @property
    def n_outputs(self):
        return self.y.shape[1]


def create_dataloaders(sequences_dir, batch_size=32, num_workers=0):
    """
    创建训练/验证/测试 DataLoader

    Args:
        sequences_dir: 序列文件目录
        batch_size: 批次大小
        num_workers: 数据加载线程数

    Returns:
        dict: {train, val, test} DataLoader
    """
    from torch.utils.data import DataLoader

    sequences_dir = Path(sequences_dir)
    loaders = {}

    for split in ['train', 'val', 'test']:
        X_path = sequences_dir / f"X_{split}.npy"
        y_path = sequences_dir / f"y_{split}.npy"

        if not X_path.exists() or not y_path.exists():
            raise FileNotFoundError(f"序列文件不存在: {X_path} 或 {y_path}")

        dataset = AQISequenceDataset(X_path, y_path)
        shuffle = (split == 'train')

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == 'train'),
        )

    return loaders
