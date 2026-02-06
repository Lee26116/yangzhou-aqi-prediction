# -*- coding: utf-8 -*-
"""
模型评估
逐 horizon 计算 MAE, RMSE, R², MAPE
生成与 XGBoost 的对比报告
提取 attention weights 样本
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from deep_learning.dl_config import (
    DL_SEQUENCES_DIR, DL_MODELS_DIR, PREDICTION_HORIZONS,
    MODEL_CONFIG, DASHBOARD_DATA_DIR
)
from deep_learning.data.dataset import create_dataloaders
from deep_learning.models.lstm_attention import build_model


def load_best_model(device):
    """加载最佳模型"""
    checkpoint_path = DL_MODELS_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    input_size = checkpoint['input_size']

    model = build_model(input_size, checkpoint.get('model_config', MODEL_CONFIG))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def evaluate_model(model, loader, device):
    """在数据集上评估模型"""
    all_preds = []
    all_targets = []
    all_attn_weights = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            predictions, attn_weights = model(X_batch)

            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())
            all_attn_weights.append(attn_weights.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    attn = np.concatenate(all_attn_weights, axis=0)

    return preds, targets, attn


def compute_metrics(preds, targets, horizons):
    """计算逐 horizon 的评估指标"""
    metrics = {}

    for i, h in enumerate(horizons):
        y_true = targets[:, i]
        y_pred = preds[:, i]

        # 过滤掉 NaN 和零值
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true > 0)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            continue

        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        r2 = r2_score(y_true_clean, y_pred_clean)
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100

        metrics[f'{h}h'] = {
            'MAE': round(float(mae), 2),
            'RMSE': round(float(rmse), 2),
            'R2': round(float(r2), 4),
            'MAPE': round(float(mape), 2),
            'n_samples': int(len(y_true_clean)),
        }

    return metrics


def load_xgboost_metrics():
    """加载 XGBoost 基线指标（按 horizon 返回）"""
    # XGBoost v4 各 horizon 的已知评估结果
    xgb_per_horizon = {
        '1h': {'MAE': 6.03, 'RMSE': 9.84, 'R2': 0.86},
        '6h': {'MAE': 11.41, 'RMSE': 17.06, 'R2': 0.71},
        '12h': {'MAE': 17.72, 'RMSE': 26.25, 'R2': 0.30},
        '24h': {'MAE': 23.24, 'RMSE': 33.79, 'R2': -0.35},
    }
    return xgb_per_horizon


def generate_comparison_report(dl_metrics, xgb_data):
    """生成 XGBoost vs LSTM 对比报告"""
    comparison = {
        'horizons': {},
        'summary': {},
    }

    for horizon_key, dl_m in dl_metrics.items():
        entry = {
            'LSTM_Attention': dl_m,
            'XGBoost': None,
            'improvement': {},
        }

        # 尝试匹配 XGBoost 指标
        if xgb_data:
            xgb_metrics = xgb_data.get(horizon_key)

            if xgb_metrics:
                entry['XGBoost'] = xgb_metrics
                # 判断胜出方（R² 更高者胜）
                dl_r2 = dl_m.get('R2', 0)
                xgb_r2 = xgb_metrics.get('R2', 0)
                entry['improvement']['winner'] = 'LSTM+Attention' if dl_r2 > xgb_r2 else 'XGBoost'
                for metric in ['MAE', 'RMSE', 'R2']:
                    if metric in dl_m and metric in xgb_metrics:
                        dl_val = dl_m[metric]
                        xgb_val = xgb_metrics[metric]
                        if metric == 'R2':
                            entry['improvement'][metric] = round(dl_val - xgb_val, 4)
                        else:
                            entry['improvement'][metric] = round(xgb_val - dl_val, 2)

        comparison['horizons'][horizon_key] = entry

    # 生成 summary
    winners = [v['improvement'].get('winner') for v in comparison['horizons'].values() if v['improvement'].get('winner')]
    xgb_wins = sum(1 for w in winners if w == 'XGBoost')
    dl_wins = sum(1 for w in winners if w == 'LSTM+Attention')
    comparison['summary'] = {
        'short_term_winner': 'XGBoost' if xgb_wins > 0 else 'LSTM+Attention',
        'long_term_winner': 'LSTM+Attention' if dl_wins > 0 else 'XGBoost',
        'note': f'XGBoost胜出{xgb_wins}个horizon, LSTM+Attention胜出{dl_wins}个horizon',
    }

    return comparison


def extract_attention_samples(attn_weights, n_samples=5):
    """提取 attention weights 样本用于可视化"""
    # 选取均匀分布的样本
    indices = np.linspace(0, len(attn_weights) - 1, n_samples, dtype=int)
    samples = []

    for idx in indices:
        # 取第一个 head 的 attention（已经是平均后的）
        attn = attn_weights[idx]  # (seq_len, seq_len) 或 (n_heads, seq_len, seq_len)
        if attn.ndim == 3:
            attn = attn.mean(axis=0)  # 平均所有 heads

        # 只保留最后一行（最后时间步对所有时间步的attention）
        last_step_attn = attn[-1].tolist()  # (seq_len,)

        samples.append({
            'index': int(idx),
            'attention_last_step': last_step_attn,
        })

    return samples


def run_evaluation():
    """运行完整评估"""
    print("=" * 60)
    print("  模型评估")
    print("=" * 60)

    # 设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"\n🖥️ 使用设备: {device}")

    # 加载模型
    print("\n📦 加载最佳模型...")
    model, checkpoint = load_best_model(device)
    print(f"   来自 epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    # 加载数据
    print("\n📦 加载测试数据...")
    loaders = create_dataloaders(DL_SEQUENCES_DIR, batch_size=64)
    test_loader = loaders['test']
    print(f"   测试集: {len(test_loader.dataset)} 样本")

    # 评估
    print("\n🔍 评估中...")
    preds, targets, attn_weights = evaluate_model(model, test_loader, device)
    print(f"   预测形状: {preds.shape}")
    print(f"   Attention 形状: {attn_weights.shape}")

    # 计算指标
    dl_metrics = compute_metrics(preds, targets, PREDICTION_HORIZONS)

    print("\n📊 LSTM+Attention 评估结果:")
    print(f"   {'Horizon':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'MAPE':<10}")
    print(f"   {'-'*50}")
    for h_key, m in dl_metrics.items():
        print(f"   {h_key:<10} {m['MAE']:<10} {m['RMSE']:<10} {m['R2']:<10} {m['MAPE']:<10}")

    # 对比 XGBoost
    print("\n📋 加载 XGBoost 基线...")
    xgb_data = load_xgboost_metrics()
    comparison = generate_comparison_report(dl_metrics, xgb_data)

    # 提取 attention 样本
    print("\n🎯 提取 Attention 样本...")
    attn_samples = extract_attention_samples(attn_weights, n_samples=10)

    # 保存结果
    results = {
        'dl_metrics': dl_metrics,
        'comparison': comparison,
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': float(checkpoint['val_loss']),
    }

    # 保存评估报告
    eval_file = DL_MODELS_DIR / "dl_evaluation.json"
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 评估报告: {eval_file}")

    # 保存 attention 样本
    attn_file = DL_MODELS_DIR / "attention_samples.json"
    with open(attn_file, 'w') as f:
        json.dump(attn_samples, f, indent=2)
    print(f"   Attention 样本: {attn_file}")

    # 同时保存到 dashboard 数据目录
    comparison_file = DASHBOARD_DATA_DIR / "model_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"   对比报告: {comparison_file}")

    return results


def main():
    run_evaluation()


if __name__ == "__main__":
    main()
