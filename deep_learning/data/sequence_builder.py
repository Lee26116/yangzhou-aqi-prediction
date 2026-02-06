# -*- coding: utf-8 -*-
"""
序列生成器
滑动窗口：48小时历史 → 预测 [1h, 6h, 12h, 24h]
StandardScaler 归一化（仅 fit 训练集）
时序分割：80/10/10（不打乱顺序）
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from deep_learning.dl_config import (
    DL_FEATURES_DIR, DL_SEQUENCES_DIR,
    SEQUENCE_LENGTH, PREDICTION_HORIZONS, SEQUENCE_STRIDE, TRAIN_CONFIG
)


def load_dl_features():
    """加载DL特征数据"""
    features_file = DL_FEATURES_DIR / "dl_features.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"DL特征文件不存在: {features_file}")

    df = pd.read_csv(features_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def create_sequences(df, seq_len, horizons, stride=1):
    """
    创建滑动窗口序列

    Args:
        df: 特征 DataFrame（含 datetime 和 AQI 列）
        seq_len: 输入序列长度
        horizons: 预测时间跨度列表
        stride: 滑动步长（默认1，增大可减少样本重叠）

    Returns:
        X: (N, seq_len, n_features) 输入序列
        y: (N, len(horizons)) 目标值
        timestamps: 每个样本对应的时间戳
    """
    feature_cols = [c for c in df.columns if c not in ['datetime', 'AQI']]
    target_col = 'AQI'

    if target_col not in df.columns:
        raise ValueError("数据中没有 AQI 列作为目标变量")

    features = df[feature_cols].values
    targets = df[target_col].values
    timestamps = df['datetime'].values

    max_horizon = max(horizons)
    max_start = len(df) - seq_len - max_horizon

    if max_start <= 0:
        raise ValueError(f"数据量不足: {len(df)} 行, 需要至少 {seq_len + max_horizon + 1} 行")

    indices = list(range(0, max_start, stride))
    n_samples = len(indices)

    X = np.zeros((n_samples, seq_len, len(feature_cols)), dtype=np.float32)
    y = np.zeros((n_samples, len(horizons)), dtype=np.float32)
    ts = []

    for idx, i in enumerate(indices):
        X[idx] = features[i:i + seq_len]
        for j, h in enumerate(horizons):
            y[idx, j] = targets[i + seq_len + h - 1]
        ts.append(timestamps[i + seq_len - 1])

    return X, y, np.array(ts), feature_cols


def split_data(X, y, timestamps, train_ratio, val_ratio):
    """
    时序分割（不打乱顺序）

    Returns:
        dict: 包含 train/val/test 的 X, y, timestamps
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        'train': {
            'X': X[:train_end],
            'y': y[:train_end],
            'timestamps': timestamps[:train_end],
        },
        'val': {
            'X': X[train_end:val_end],
            'y': y[train_end:val_end],
            'timestamps': timestamps[train_end:val_end],
        },
        'test': {
            'X': X[val_end:],
            'y': y[val_end:],
            'timestamps': timestamps[val_end:],
        },
    }

    return splits


def normalize_sequences(splits):
    """
    使用 StandardScaler 归一化（仅 fit 训练集）

    Args:
        splits: dict with train/val/test X arrays

    Returns:
        归一化后的 splits, scaler
    """
    train_X = splits['train']['X']
    n_train, seq_len, n_features = train_X.shape

    # 将3D reshape为2D来fit scaler
    train_2d = train_X.reshape(-1, n_features)
    scaler = StandardScaler()
    scaler.fit(train_2d)

    # 对每个split进行归一化
    for key in ['train', 'val', 'test']:
        X = splits[key]['X']
        n, s, f = X.shape
        X_2d = X.reshape(-1, f)
        X_scaled = scaler.transform(X_2d)
        splits[key]['X'] = X_scaled.reshape(n, s, f).astype(np.float32)

    return splits, scaler


def build_sequences():
    """构建完整的序列数据"""
    print("=" * 60)
    print("  序列生成")
    print("=" * 60)

    # 加载特征
    print("\n📖 加载DL特征数据...")
    df = load_dl_features()
    print(f"   数据形状: {df.shape}")

    # 创建序列
    print(f"\n🔧 创建滑动窗口序列 (窗口={SEQUENCE_LENGTH}h, 步长={SEQUENCE_STRIDE}, 预测={PREDICTION_HORIZONS})...")
    X, y, timestamps, feature_names = create_sequences(
        df, SEQUENCE_LENGTH, PREDICTION_HORIZONS, stride=SEQUENCE_STRIDE
    )
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   特征数: {len(feature_names)}")

    # 时序分割
    train_ratio = TRAIN_CONFIG['train_ratio']
    val_ratio = TRAIN_CONFIG['val_ratio']
    print(f"\n📊 时序分割 ({train_ratio}/{val_ratio}/{TRAIN_CONFIG['test_ratio']})...")
    splits = split_data(X, y, timestamps, train_ratio, val_ratio)

    for key in ['train', 'val', 'test']:
        print(f"   {key}: X={splits[key]['X'].shape}, y={splits[key]['y'].shape}")

    # 归一化
    print("\n📏 StandardScaler 归一化（仅fit训练集）...")
    splits, scaler = normalize_sequences(splits)

    # 保存
    print(f"\n💾 保存到 {DL_SEQUENCES_DIR}...")

    for key in ['train', 'val', 'test']:
        np.save(DL_SEQUENCES_DIR / f"X_{key}.npy", splits[key]['X'])
        np.save(DL_SEQUENCES_DIR / f"y_{key}.npy", splits[key]['y'])
        np.save(DL_SEQUENCES_DIR / f"timestamps_{key}.npy", splits[key]['timestamps'])
        print(f"   {key}: X={splits[key]['X'].shape}, y={splits[key]['y'].shape}")

    # 保存 scaler
    scaler_file = DL_SEQUENCES_DIR / "scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   scaler: {scaler_file}")

    # 保存特征名
    names_file = DL_SEQUENCES_DIR / "feature_names.txt"
    with open(names_file, 'w') as f:
        f.write('\n'.join(feature_names))
    print(f"   feature_names: {names_file}")

    # 保存元数据
    import json
    meta = {
        'sequence_length': SEQUENCE_LENGTH,
        'prediction_horizons': PREDICTION_HORIZONS,
        'n_features': len(feature_names),
        'n_train': len(splits['train']['X']),
        'n_val': len(splits['val']['X']),
        'n_test': len(splits['test']['X']),
        'feature_names': feature_names,
    }
    meta_file = DL_SEQUENCES_DIR / "metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ 序列生成完成!")
    print(f"   总样本数: {len(X)}")
    print(f"   N ≈ {len(X)}")

    return splits, scaler, feature_names


def main():
    build_sequences()


if __name__ == "__main__":
    main()
