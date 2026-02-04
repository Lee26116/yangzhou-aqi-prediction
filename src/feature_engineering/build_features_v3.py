# -*- coding: utf-8 -*-
"""
特征工程 v3 - 方法论正确的版本

核心原则：
1. 预测 h 小时后的 AQI，只能使用 h 小时前及更早的数据
2. 训练时和预测时使用完全相同的信息
3. 不依赖滚动预测，避免误差累积

例如预测 24h 后：
- 可以用：当前 AQI（即 lag_24h）、48h 前的 AQI（即 lag_48h）
- 不能用：lag_1h, lag_3h（预测时这些信息不可知）
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DATA_DIR, FEATURES_DATA_DIR


def build_features_for_horizon(df, horizon_hours):
    """
    为特定预测时间跨度构建特征

    Args:
        df: 原始数据（包含 datetime, AQI, 气象等）
        horizon_hours: 预测时间跨度（小时）

    Returns:
        DataFrame: 包含特征和目标变量

    关键：所有特征都是相对于"预测时刻"而言的历史数据
    """
    print(f"\n🔧 构建 {horizon_hours}h 预测特征...")

    result = df[['datetime', 'AQI']].copy()
    result = result.rename(columns={'AQI': 'AQI_target'})  # 这是我们要预测的目标

    # ========== 1. AQI 滞后特征 ==========
    # 预测 h 小时后，最近可用的 AQI 是 h 小时前的值
    # 所以 "lag_0" 实际上对应原始数据的 shift(horizon_hours)

    print(f"   1. AQI 滞后特征（相对于预测时刻 -{horizon_hours}h 起算）...")

    # 基础滞后（从预测时刻往前算）
    lag_offsets = [0, 1, 3, 6, 12, 24, 48]  # 相对于预测时刻的滞后
    for offset in lag_offsets:
        actual_lag = horizon_hours + offset
        result[f'AQI_lag_{offset}h'] = df['AQI'].shift(actual_lag)

    # ========== 2. AQI 滚动统计（基于预测时刻可知的数据）==========
    print(f"   2. AQI 滚动统计...")

    # 24h 滚动统计（从 lag_0 往前算 24 小时）
    # 这意味着实际是从 horizon_hours 往前算 24 小时
    for window in [24, 48]:
        # 先 shift(horizon_hours)，再算滚动统计
        shifted_aqi = df['AQI'].shift(horizon_hours)
        result[f'AQI_rolling_mean_{window}h'] = shifted_aqi.rolling(window=window, min_periods=1).mean()
        result[f'AQI_rolling_std_{window}h'] = shifted_aqi.rolling(window=window, min_periods=1).std()
        result[f'AQI_rolling_max_{window}h'] = shifted_aqi.rolling(window=window, min_periods=1).max()
        result[f'AQI_rolling_min_{window}h'] = shifted_aqi.rolling(window=window, min_periods=1).min()

    # ========== 3. AQI 变化趋势（基于历史数据）==========
    print(f"   3. AQI 变化趋势...")

    # 过去 24h 的变化（lag_0 相比 lag_24）
    result['AQI_change_24h'] = result['AQI_lag_0h'] - result['AQI_lag_24h']
    result['AQI_change_48h'] = result['AQI_lag_0h'] - result['AQI_lag_48h']

    # ========== 4. 历史同期特征 ==========
    print(f"   4. 历史同期特征...")

    # 昨天同一时刻的 AQI
    result['AQI_yesterday_same_hour'] = df['AQI'].shift(24 + horizon_hours)
    # 前天同一时刻
    result['AQI_2days_ago_same_hour'] = df['AQI'].shift(48 + horizon_hours)
    # 一周前同一时刻
    result['AQI_week_ago_same_hour'] = df['AQI'].shift(168 + horizon_hours)

    # ========== 5. PM2.5/PM10 特征（如果有）==========
    print(f"   5. PM2.5/PM10 特征...")

    for pollutant in ['PM2.5', 'PM10']:
        if pollutant in df.columns:
            # 滞后特征
            result[f'{pollutant}_lag_0h'] = df[pollutant].shift(horizon_hours)
            result[f'{pollutant}_lag_24h'] = df[pollutant].shift(horizon_hours + 24)

            # 滚动统计
            shifted = df[pollutant].shift(horizon_hours)
            result[f'{pollutant}_rolling_mean_24h'] = shifted.rolling(24, min_periods=1).mean()

    # ========== 6. 时间特征（预测目标时刻的）==========
    print(f"   6. 时间特征...")

    # 这些是预测目标时刻的时间特征，在预测时是已知的
    result['hour'] = df['datetime'].dt.hour
    result['day_of_week'] = df['datetime'].dt.dayofweek
    result['month'] = df['datetime'].dt.month
    result['day_of_year'] = df['datetime'].dt.dayofyear

    # 周期编码
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
    result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
    result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
    result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)

    # 特殊时段标记
    result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
    result['is_rush_hour'] = (
        ((result['hour'] >= 7) & (result['hour'] <= 9)) |
        ((result['hour'] >= 17) & (result['hour'] <= 19))
    ).astype(int)
    result['is_night'] = ((result['hour'] >= 22) | (result['hour'] <= 5)).astype(int)

    # ========== 7. 气象特征（如果有天气预报数据）==========
    # 注意：理想情况下应该用天气预报数据
    # 这里暂时用历史气象的统计特征作为代理
    print(f"   7. 气象特征...")

    weather_cols = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
                    'surface_pressure', 'dew_point_2m']

    for col in weather_cols:
        if col in df.columns:
            # 历史同期气象（昨天同一时刻）
            result[f'{col}_yesterday'] = df[col].shift(24 + horizon_hours)
            # 最近可知的气象（horizon_hours 前）
            result[f'{col}_lag_0h'] = df[col].shift(horizon_hours)

    # ========== 8. 季节性代理变量 ==========
    print(f"   8. 季节性代理变量...")

    # 供暖季（11月-3月）
    result['is_heating_season'] = result['month'].isin([11, 12, 1, 2, 3]).astype(int)

    # 春节期间（简化：1月下旬-2月中旬）
    day = df['datetime'].dt.day
    result['is_spring_festival'] = (
        ((result['month'] == 1) & (day >= 20)) |
        ((result['month'] == 2) & (day <= 15))
    ).astype(int)

    # 清除不需要的临时列
    result = result.drop(columns=['hour', 'day_of_week', 'month', 'day_of_year'], errors='ignore')

    # 统计
    feature_cols = [c for c in result.columns if c not in ['datetime', 'AQI_target']]
    print(f"   ✅ 特征构建完成: {len(feature_cols)} 个特征")

    return result


def select_features(df, target='AQI_target', min_corr=0.05, max_features=30):
    """
    特征选择
    """
    print(f"\n🎯 特征选择...")

    feature_cols = [c for c in df.columns if c not in ['datetime', 'AQI_target']]

    # 排除缺失率过高的特征
    missing_rate = df[feature_cols].isnull().mean()
    valid_features = missing_rate[missing_rate < 0.3].index.tolist()
    print(f"   缺失率筛选后: {len(valid_features)} 个")

    # 计算与目标的相关性
    valid_df = df.dropna(subset=[target])
    correlations = valid_df[valid_features].corrwith(valid_df[target]).abs()
    correlations = correlations.dropna().sort_values(ascending=False)

    # 筛选相关性足够的特征
    selected = correlations[correlations >= min_corr].index.tolist()
    print(f"   相关性筛选后 (>={min_corr}): {len(selected)} 个")

    # 限制数量
    if len(selected) > max_features:
        selected = selected[:max_features]

    print(f"   最终选择: {len(selected)} 个特征")
    print(f"\n   Top 10 特征:")
    for i, feat in enumerate(selected[:10]):
        print(f"      {i+1}. {feat}: {correlations[feat]:.4f}")

    return selected


def main():
    """主函数"""
    print("=" * 60)
    print("  特征工程 v3 - 方法论正确版本")
    print("  原则: 预测 h 小时后，只使用 h 小时前及更早的数据")
    print("=" * 60)

    # 读取数据
    input_file = PROCESSED_DATA_DIR / "yangzhou_merged.csv"
    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return

    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"\n📂 读取数据: {len(df)} 条")

    # 为不同预测时间跨度构建特征
    horizons = [1, 6, 12, 24]

    for h in horizons:
        print(f"\n{'='*60}")
        print(f"  预测时间跨度: {h} 小时")
        print(f"{'='*60}")

        # 构建特征
        features_df = build_features_for_horizon(df, h)

        # 删除缺失目标的行
        features_df = features_df.dropna(subset=['AQI_target'])
        print(f"\n   有效样本数: {len(features_df)}")

        # 特征选择
        selected = select_features(features_df)

        # 保存
        output_cols = ['datetime', 'AQI_target'] + selected
        output_cols = [c for c in output_cols if c in features_df.columns]

        output_file = FEATURES_DATA_DIR / f"yangzhou_features_v3_{h}h.csv"
        features_df[output_cols].to_csv(output_file, index=False)
        print(f"\n💾 已保存: {output_file}")

    print("\n" + "=" * 60)
    print("  特征工程 v3 完成")
    print("  生成了 4 个数据集，分别用于预测 1h, 6h, 12h, 24h")
    print("=" * 60)


if __name__ == "__main__":
    main()
