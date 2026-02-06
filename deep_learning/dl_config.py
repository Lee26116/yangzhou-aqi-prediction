# -*- coding: utf-8 -*-
"""
深度学习模块配置和超参数
"""

import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR,
    MODELS_DIR, DASHBOARD_DATA_DIR, YANGZHOU_CONFIG
)

# ==================== 目录配置 ====================

DL_DIR = PROJECT_ROOT / "deep_learning"
DL_FEATURES_DIR = FEATURES_DATA_DIR / "dl_features"
DL_SEQUENCES_DIR = FEATURES_DATA_DIR / "dl_sequences"
DL_MODELS_DIR = MODELS_DIR / "deep_learning"
DL_EXPORTED_DIR = DL_MODELS_DIR / "exported"

# 确保目录存在
for d in [DL_FEATURES_DIR, DL_SEQUENCES_DIR, DL_MODELS_DIR, DL_EXPORTED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==================== 序列参数 ====================

SEQUENCE_LENGTH = 48          # 输入序列长度（小时）
PREDICTION_HORIZONS = [1, 6, 12, 24]  # 预测时间跨度（小时）
SEQUENCE_STRIDE = 6           # 滑动窗口步长（减少样本重叠）

# ==================== 模型架构 ====================

MODEL_CONFIG = {
    "input_size": None,       # 运行时由特征数决定
    "hidden_size": 64,
    "num_layers": 1,
    "num_heads": 4,
    "dropout": 0.3,
    "bidirectional": True,
    "fc_sizes": [64],
    "output_size": len(PREDICTION_HORIZONS),  # 4
}

# ==================== 训练超参数 ====================

TRAIN_CONFIG = {
    "batch_size": 64,
    "max_epochs": 300,
    "learning_rate": 0.001,
    "weight_decay": 1e-3,
    "gradient_clip_max_norm": 1.0,
    "early_stopping_patience": 40,
    "noise_std": 0.05,       # 训练时输入高斯噪声
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    # 加权 Huber Loss 权重（短期优先）
    "horizon_weights": [1.0, 0.8, 0.6, 0.4],
    "huber_delta": 1.0,
}

# ==================== 上风向城市空间滞后配置 ====================

SPATIAL_LAG_CONFIG = {
    "南京": {
        "lags": [1, 3, 6, 12],
        "pm25_lags": [3, 6],
    },
    "镇江": {
        "lags": [1, 3, 6],
        "pm25_lags": [],
    },
    "泰州": {
        "lags": [1, 3, 6],
        "pm25_lags": [],
    },
    "南通": {
        "lags": [1, 3],
        "pm25_lags": [],
    },
}

# 城市到扬州的距离（km，用于风向感知权重）
CITY_DISTANCES = {
    "南京": 80,
    "镇江": 30,
    "泰州": 50,
    "南通": 130,
}

# 城市相对于扬州的方位角（度，北=0，顺时针）
CITY_BEARINGS = {
    "南京": 250,    # 西南
    "镇江": 190,    # 南
    "泰州": 80,     # 东
    "南通": 130,    # 东南
}

# ==================== ONNX 导出 ====================

ONNX_CONFIG = {
    "opset_version": 17,
    "model_filename": "model.onnx",
}

# ==================== 性能基准 ====================

BENCHMARK_THRESHOLDS = {
    "inference_time_seconds": 1.0,
    "memory_mb": 1024,
    "model_size_mb": 50,
}
