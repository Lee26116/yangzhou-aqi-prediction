# -*- coding: utf-8 -*-
"""
深度学习 Dashboard 数据生成
生成 dl_overview.json, model_comparison.json, training_curves.json, attention_samples.json
"""

import sys
import json
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DASHBOARD_DATA_DIR, get_aqi_level, get_china_now
from deep_learning.dl_config import (
    DL_MODELS_DIR, DL_SEQUENCES_DIR, DL_FEATURES_DIR, PREDICTION_HORIZONS
)


def generate_training_curves():
    """生成训练/验证 loss 曲线数据"""
    print("📈 生成训练曲线数据...")

    history_file = DL_MODELS_DIR / "training_history.json"
    if not history_file.exists():
        print("   ⚠️ 训练历史不存在，跳过")
        return None

    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)

    curves = {
        'epochs': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'learning_rate': history.get('lr', []),
        'best_epoch': None,
    }

    # 标记最佳 epoch
    if history['val_loss']:
        best_idx = int(np.argmin(history['val_loss']))
        curves['best_epoch'] = best_idx + 1
        curves['best_val_loss'] = history['val_loss'][best_idx]

    output_file = DASHBOARD_DATA_DIR / "training_curves.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(curves, f, ensure_ascii=False, indent=2, default=str)

    print(f"   ✅ 训练曲线: {output_file}")
    return curves


def generate_attention_data():
    """生成 Attention 热力图数据"""
    print("🎯 生成 Attention 数据...")

    attn_file = DL_MODELS_DIR / "attention_samples.json"
    if not attn_file.exists():
        print("   ⚠️ Attention 样本不存在，跳过")
        return None

    with open(attn_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # 加载特征名
    feature_names_file = DL_SEQUENCES_DIR / "feature_names.txt"
    feature_names = []
    if feature_names_file.exists():
        with open(feature_names_file, 'r', encoding='utf-8') as f:
            feature_names = [line.strip() for line in f.readlines()]

    # 加载元数据
    meta_file = DL_SEQUENCES_DIR / "metadata.json"
    seq_len = 48
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            seq_len = meta.get('sequence_length', 48)

    attention_data = {
        'sequence_length': seq_len,
        'time_labels': [f't-{seq_len - i}h' for i in range(seq_len)],
        'feature_names': feature_names,
        'samples': samples,
    }

    output_file = DASHBOARD_DATA_DIR / "attention_samples.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(attention_data, f, ensure_ascii=False, indent=2, default=str)

    print(f"   ✅ Attention 数据: {output_file}")
    return attention_data


def generate_model_comparison():
    """生成模型对比数据"""
    print("📊 生成模型对比数据...")

    # 检查是否已在评估阶段生成
    comparison_file = DASHBOARD_DATA_DIR / "model_comparison.json"
    if comparison_file.exists():
        print(f"   ✅ 对比数据已存在: {comparison_file}")
        return

    # 尝试从评估结果生成
    eval_file = DL_MODELS_DIR / "dl_evaluation.json"
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)

        comparison = eval_data.get('comparison', {})
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        print(f"   ✅ 对比数据: {comparison_file}")
    else:
        print("   ⚠️ 评估结果不存在，跳过")


def generate_dl_overview():
    """生成深度学习模型的预测概览数据 (dl_overview.json)"""
    print("🔮 生成 DL 预测概览...")

    from deep_learning.inference.predict_dl import get_dl_predictor

    # 加载 DL 特征
    dl_features_file = DL_FEATURES_DIR / "dl_features.csv"
    if not dl_features_file.exists():
        print(f"   ⚠️ DL 特征文件不存在: {dl_features_file}")
        return None

    df = pd.read_csv(dl_features_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 获取最近72小时数据
    latest_time = df['datetime'].max()
    start_time = latest_time - timedelta(hours=72)
    recent_data = df[df['datetime'] >= start_time].copy()

    if len(recent_data) < 48:
        print(f"   ⚠️ 数据不足 (仅 {len(recent_data)} 行, 需要 48)")
        return None

    # 加载预测器
    try:
        predictor = get_dl_predictor()
        predictor.load()
    except Exception as e:
        print(f"   ⚠️ DL 模型加载失败: {e}")
        return None

    # 获取实时 AQI
    realtime = None
    try:
        import requests
        from src.config import WAQI_TOKEN
        url = f"https://api.waqi.info/feed/yangzhou/?token={WAQI_TOKEN}"
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get('status') == 'ok':
            aqi_data = data['data']
            iaqi = aqi_data.get('iaqi', {})
            def get_iaqi_value(key):
                val = iaqi.get(key)
                return val.get('v') if isinstance(val, dict) else val
            realtime = {
                'aqi': aqi_data.get('aqi'),
                'pm25': get_iaqi_value('pm25'),
                'pm10': get_iaqi_value('pm10'),
                'temperature': get_iaqi_value('t'),
                'humidity': get_iaqi_value('h'),
                'time': aqi_data.get('time', {}).get('iso'),
                'dominant_pollutant': aqi_data.get('dominentpol'),
            }
    except Exception as e:
        print(f"   ⚠️ 获取实时数据失败: {e}")

    # MC Dropout 预测（含置信区间）
    try:
        uncertainty_dict = predictor.predict_with_uncertainty(recent_data, n_samples=50)
        predictions_dict = {h: v['mean'] for h, v in uncertainty_dict.items()}
    except Exception as e:
        print(f"   ⚠️ DL 预测失败: {e}")
        return None

    # 构建预测列表（4个关键点之间插值到24小时）
    china_now = get_china_now()
    current_time = china_now.replace(minute=0, second=0, microsecond=0)

    dl_predictions = []
    for h in range(1, 25):
        pred_time = current_time + timedelta(hours=h)

        if h in predictions_dict:
            pred_aqi = predictions_dict[h]
            pred_lower = uncertainty_dict[h]['lower']
            pred_upper = uncertainty_dict[h]['upper']
        else:
            lower_h = max(k for k in PREDICTION_HORIZONS if k <= h)
            upper_h = min(k for k in PREDICTION_HORIZONS if k >= h)
            if lower_h == upper_h:
                pred_aqi = predictions_dict[lower_h]
                pred_lower = uncertainty_dict[lower_h]['lower']
                pred_upper = uncertainty_dict[lower_h]['upper']
            else:
                ratio = (h - lower_h) / (upper_h - lower_h)
                pred_aqi = predictions_dict[lower_h] * (1 - ratio) + predictions_dict[upper_h] * ratio
                pred_lower = uncertainty_dict[lower_h]['lower'] * (1 - ratio) + uncertainty_dict[upper_h]['lower'] * ratio
                pred_upper = uncertainty_dict[lower_h]['upper'] * (1 - ratio) + uncertainty_dict[upper_h]['upper'] * ratio

        pred_aqi = max(10, pred_aqi)
        pred_lower = max(5, pred_lower)
        pred_upper = max(pred_aqi, pred_upper)
        level, color = get_aqi_level(pred_aqi)

        dl_predictions.append({
            'datetime': pred_time.isoformat(),
            'hour': h,
            'predicted_aqi': round(pred_aqi, 1),
            'lower_bound': round(pred_lower, 1),
            'upper_bound': round(pred_upper, 1),
            'level': level,
            'color': color,
        })

    dl_overview = {
        'update_time': china_now.isoformat(),
        'update_time_display': china_now.strftime('%Y-%m-%d %H:%M:%S') + ' (北京时间)',
        'timezone': 'Asia/Shanghai (UTC+8)',
        'model': 'LSTM+Attention',
        'realtime': realtime,
        'predictions': dl_predictions,
        'key_predictions': {f'{h}h': predictions_dict.get(h) for h in PREDICTION_HORIZONS},
        'historical': {
            'last_24h': recent_data[['datetime', 'AQI']].tail(24).to_dict(orient='records')
            if 'AQI' in recent_data.columns else []
        }
    }

    overview_file = DASHBOARD_DATA_DIR / "dl_overview.json"
    with open(overview_file, 'w', encoding='utf-8') as f:
        json.dump(dl_overview, f, ensure_ascii=False, indent=2, default=str)

    print(f"   ✅ DL 预测概览: {overview_file}")
    return dl_overview


def generate_all():
    """生成所有 Dashboard 数据"""
    print("=" * 60)
    print("  深度学习 Dashboard 数据生成")
    print("=" * 60)

    generate_dl_overview()
    generate_training_curves()
    generate_attention_data()
    generate_model_comparison()

    print(f"\n✅ 所有 Dashboard 数据已生成到: {DASHBOARD_DATA_DIR}")


def main():
    generate_all()


if __name__ == "__main__":
    main()
