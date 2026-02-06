# -*- coding: utf-8 -*-
"""
深度学习推理模块
使用 ONNX 模型进行 AQI 预测
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from deep_learning.dl_config import (
    DL_EXPORTED_DIR, DL_SEQUENCES_DIR, DL_FEATURES_DIR,
    SEQUENCE_LENGTH, PREDICTION_HORIZONS, ONNX_CONFIG
)
from src.config import get_aqi_level


class DLPredictor:
    """深度学习 AQI 预测器"""

    def __init__(self):
        self.session = None
        self.scaler = None
        self.feature_names = None
        self.input_name = None
        self._loaded = False

    def load(self):
        """加载 ONNX 模型 + scaler + feature_names"""
        if self._loaded:
            return

        # 加载 ONNX 模型
        onnx_path = DL_EXPORTED_DIR / ONNX_CONFIG['model_filename']
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX 模型不存在: {onnx_path}")

        import onnxruntime as ort
        self.session = ort.InferenceSession(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name

        # 加载 scaler
        scaler_path = DL_SEQUENCES_DIR / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler 不存在: {scaler_path}")

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # 加载特征名
        names_path = DL_SEQUENCES_DIR / "feature_names.txt"
        if not names_path.exists():
            raise FileNotFoundError(f"特征名文件不存在: {names_path}")

        with open(names_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]

        self._loaded = True
        print(f"   ✅ DL模型加载完成 (特征数: {len(self.feature_names)})")

    def prepare_input(self, recent_data):
        """
        从最近48小时数据构建输入序列

        Args:
            recent_data: DataFrame，至少含有 SEQUENCE_LENGTH 行

        Returns:
            np.ndarray: (1, seq_len, n_features) 归一化后的输入
        """
        if not self._loaded:
            self.load()

        # 确保有足够数据
        if len(recent_data) < SEQUENCE_LENGTH:
            raise ValueError(
                f"数据不足: 需要 {SEQUENCE_LENGTH} 行, 只有 {len(recent_data)} 行"
            )

        # 选择最后 SEQUENCE_LENGTH 行
        data = recent_data.tail(SEQUENCE_LENGTH)

        # 提取特征列（按顺序）
        missing_features = []
        feature_values = []
        for feat in self.feature_names:
            if feat in data.columns:
                feature_values.append(data[feat].values)
            else:
                # 用0填充缺失特征
                feature_values.append(np.zeros(SEQUENCE_LENGTH))
                missing_features.append(feat)

        if missing_features:
            print(f"   ⚠️ 缺失 {len(missing_features)} 个特征(零填充): {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")

        X = np.column_stack(feature_values).astype(np.float32)  # (seq_len, n_features)

        # 处理 NaN 和 Inf
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"   ⚠️ 输入数据含 {nan_count} 个NaN, {inf_count} 个Inf, 已替换为0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 归一化
        X_scaled = self.scaler.transform(X)

        # 添加 batch 维度
        X_batch = X_scaled[np.newaxis, :, :].astype(np.float32)  # (1, seq_len, n_features)

        return X_batch

    def predict(self, X_batch):
        """
        使用 ONNX 推理

        Args:
            X_batch: (batch, seq_len, n_features)

        Returns:
            dict: {horizon: predicted_aqi} e.g. {1: 65.2, 6: 70.1, 12: 75.3, 24: 80.0}
        """
        if not self._loaded:
            self.load()

        outputs = self.session.run(None, {self.input_name: X_batch})
        predictions = outputs[0][0]  # (n_horizons,)

        result = {}
        for i, h in enumerate(PREDICTION_HORIZONS):
            aqi = float(max(0, predictions[i]))
            result[h] = round(aqi, 1)

        return result

    def predict_from_dataframe(self, df):
        """
        从 DataFrame 直接预测

        Args:
            df: 含有至少 SEQUENCE_LENGTH 行的 DataFrame

        Returns:
            dict: horizon -> AQI 预测值
        """
        X_batch = self.prepare_input(df)
        return self.predict(X_batch)

    def _load_pytorch_model(self):
        """加载 PyTorch 模型（用于 MC Dropout 和 Attention）"""
        import torch
        from deep_learning.dl_config import DL_MODELS_DIR, MODEL_CONFIG
        from deep_learning.models.lstm_attention import build_model

        checkpoint_path = DL_MODELS_DIR / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"模型不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model = build_model(checkpoint['input_size'], checkpoint.get('model_config', MODEL_CONFIG))
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def predict_with_uncertainty(self, recent_data, n_samples=50):
        """
        MC Dropout 不确定性估计

        在 train 模式下多次前向传播，利用 Dropout 随机性估计预测不确定性。

        Args:
            recent_data: DataFrame
            n_samples: MC 采样次数

        Returns:
            dict: {horizon: {'mean': float, 'lower': float, 'upper': float}}
        """
        try:
            import torch

            model = self._load_pytorch_model()
            model.cpu()
            model.train()  # 保持 Dropout 激活，用于 MC 不确定性估计

            X_batch = self.prepare_input(recent_data)
            X_tensor = torch.from_numpy(X_batch).cpu()

            all_preds = []
            with torch.no_grad():
                for _ in range(n_samples):
                    preds, _ = model(X_tensor)
                    all_preds.append(preds[0].detach().cpu().numpy())

            all_preds = np.array(all_preds)  # (n_samples, n_horizons)

            result = {}
            for i, h in enumerate(PREDICTION_HORIZONS):
                values = all_preds[:, i]
                result[h] = {
                    'mean': round(float(max(0, np.mean(values))), 1),
                    'lower': round(float(max(0, np.percentile(values, 2.5))), 1),
                    'upper': round(float(max(0, np.percentile(values, 97.5))), 1),
                    'std': round(float(np.std(values)), 2),
                }

            return result

        except Exception as e:
            print(f"   ⚠️ MC Dropout 失败: {e}")
            # 回退到确定性预测
            preds = self.predict_from_dataframe(recent_data)
            return {h: {'mean': v, 'lower': v * 0.85, 'upper': v * 1.15, 'std': 0.0}
                    for h, v in preds.items()}

    def predict_with_attention(self, recent_data):
        """
        使用 PyTorch 模型获取 attention 权重（用于可视化）

        Args:
            recent_data: DataFrame

        Returns:
            tuple: (predictions_dict, attention_weights)
        """
        try:
            import torch

            model = self._load_pytorch_model()
            model.cpu()
            model.eval()  # 确定性模式，用于获取稳定的 attention 权重

            X_batch = self.prepare_input(recent_data)
            X_tensor = torch.from_numpy(X_batch).cpu()

            with torch.no_grad():
                preds, attn_weights = model(X_tensor)

            predictions = {}
            for i, h in enumerate(PREDICTION_HORIZONS):
                predictions[h] = round(float(max(0, preds[0][i].detach().cpu().item())), 1)

            attn = attn_weights[0].detach().cpu().numpy()  # (seq_len, seq_len)
            if attn.ndim == 3:
                attn = attn.mean(axis=0)

            return predictions, attn

        except Exception as e:
            print(f"   ⚠️ Attention 获取失败: {e}")
            return self.predict_from_dataframe(recent_data), None


def get_dl_predictor():
    """获取全局 DLPredictor 实例"""
    if not hasattr(get_dl_predictor, '_instance'):
        get_dl_predictor._instance = DLPredictor()
    return get_dl_predictor._instance
