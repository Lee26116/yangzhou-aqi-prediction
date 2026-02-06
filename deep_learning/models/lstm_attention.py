# -*- coding: utf-8 -*-
"""
LSTM + Multi-Head Self-Attention 模型
用于多时间跨度 AQI 预测

实际训练架构 (dl_config.py MODEL_CONFIG):
  Input (batch, 48, 47)
    → BiLSTM Encoder (hidden=64, layers=1, bidirectional=True, dropout=0.3)
    → Output (batch, 48, 128)
    → Multi-Head Self-Attention (heads=4, embed_dim=128)
    → Residual Connection + LayerNorm
    → 取最后时间步 context (batch, 128)
    → FC: 128→64→ReLU→Dropout(0.3)→4
    → Output (batch, 4) = [AQI_1h, AQI_6h, AQI_12h, AQI_24h]

注: __init__ 的默认参数 (hidden=128, layers=2) 是较大配置，
    实际通过 dl_config.py 传入较小值 (hidden=64, layers=1, ~132K参数)。
"""

import torch
import torch.nn as nn
import math


class AQIPredictor(nn.Module):
    """LSTM+Attention AQI 预测模型（训练用，返回 attention weights）"""

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 num_heads=4, dropout=0.2, bidirectional=True,
                 fc_sizes=None, output_size=4):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm_output_size = hidden_size * self.num_directions  # 64*2=128 (实际配置)

        if fc_sizes is None:
            fc_sizes = [128, 64]

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_output_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(self.lstm_output_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # FC Decoder
        fc_layers = []
        prev_size = self.lstm_output_size
        for fc_size in fc_sizes:
            fc_layers.extend([
                nn.Linear(prev_size, fc_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = fc_size
        fc_layers.append(nn.Linear(prev_size, output_size))

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            predictions: (batch, output_size)
            attention_weights: (batch, seq_len, seq_len)
        """
        # LSTM Encoder
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_output_size)

        # Multi-Head Self-Attention with residual connection
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )  # (batch, seq_len, lstm_output_size), (batch, seq_len, seq_len)

        # Residual + LayerNorm
        out = self.layer_norm(lstm_out + attn_out)
        out = self.dropout(out)

        # 取最后时间步作为 context
        context = out[:, -1, :]  # (batch, lstm_output_size)

        # FC Decoder
        predictions = self.fc(context)  # (batch, output_size)

        return predictions, attn_weights


class AQIPredictorInference(nn.Module):
    """推理专用模型（不返回 attention weights，用于 ONNX 导出）"""

    def __init__(self, model):
        """
        Args:
            model: 训练好的 AQIPredictor 实例
        """
        super().__init__()
        self.lstm = model.lstm
        self.attention = model.attention
        self.layer_norm = model.layer_norm
        self.dropout = model.dropout
        self.fc = model.fc

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            predictions: (batch, output_size)
        """
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.layer_norm(lstm_out + attn_out)
        out = self.dropout(out)
        # 使用 select 替代 [:, -1, :] 以兼容 ONNX 导出
        context = torch.select(out, 1, out.size(1) - 1)
        predictions = self.fc(context)
        return predictions


class WeightedHuberLoss(nn.Module):
    """加权 Huber Loss，短期预测权重更高"""

    def __init__(self, weights=None, delta=1.0):
        super().__init__()
        if weights is None:
            weights = [1.0, 0.8, 0.6, 0.4]
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.delta = delta

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch, n_horizons)
            targets: (batch, n_horizons)

        Returns:
            weighted loss scalar
        """
        huber = nn.functional.huber_loss(
            predictions, targets, reduction='none', delta=self.delta
        )  # (batch, n_horizons)

        weighted = huber * self.weights.unsqueeze(0)  # broadcast weights
        return weighted.mean()


def build_model(input_size, config=None):
    """
    构建模型

    Args:
        input_size: 输入特征数
        config: 模型配置字典

    Returns:
        AQIPredictor 模型
    """
    if config is None:
        from deep_learning.dl_config import MODEL_CONFIG
        config = MODEL_CONFIG

    model = AQIPredictor(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
        fc_sizes=config['fc_sizes'],
        output_size=config['output_size'],
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   模型参数: {total_params:,} (可训练: {trainable_params:,})")

    return model
