# -*- coding: utf-8 -*-
"""
特征工程 v2 - 不使用其他城市同期AQI
专注于本地数据、气象、时间特征的预测能力
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DATA_DIR, FEATURES_DATA_DIR


def build_lag_features(df, col, lags):
    """构建滞后特征"""
    for lag in lags:
        df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
    return df


def build_rolling_features(df, col, windows):
    """构建滚动统计特征"""
    for window in windows:
        df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window=window, min_periods=1).mean()
        df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window=window, min_periods=1).std()
        df[f'{col}_rolling_max_{window}h'] = df[col].rolling(window=window, min_periods=1).max()
        df[f'{col}_rolling_min_{window}h'] = df[col].rolling(window=window, min_periods=1).min()
    return df


def build_change_features(df, col, periods):
    """构建变化率特征"""
    for period in periods:
        df[f'{col}_change_{period}h'] = df[col] - df[col].shift(period)
        df[f'{col}_pct_change_{period}h'] = df[col].pct_change(periods=period) * 100
    return df


def build_all_features(df):
    """
    构建所有特征（不使用其他城市同期AQI）
    """
    print("🔧 特征工程 v2 - 构建预测特征...")

    df = df.copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    # ========== 1. AQI 滞后特征 ==========
    print("   1. AQI 滞后特征...")
    aqi_lags = [1, 2, 3, 6, 12, 24, 48]
    df = build_lag_features(df, 'AQI', aqi_lags)

    # ========== 2. AQI 滚动统计 ==========
    print("   2. AQI 滚动统计...")
    aqi_windows = [3, 6, 12, 24]
    df = build_rolling_features(df, 'AQI', aqi_windows)

    # ========== 3. AQI 变化率 ==========
    print("   3. AQI 变化率...")
    aqi_periods = [1, 3, 6, 12, 24]
    df = build_change_features(df, 'AQI', aqi_periods)

    # ========== 4. PM2.5 特征 ==========
    print("   4. PM2.5 特征...")
    if 'PM2.5' in df.columns:
        pm25_lags = [1, 3, 6, 24]
        df = build_lag_features(df, 'PM2.5', pm25_lags)
        df = build_rolling_features(df, 'PM2.5', [6, 24])
        df = build_change_features(df, 'PM2.5', [1, 3, 6])

    # ========== 5. PM10 特征 ==========
    print("   5. PM10 特征...")
    if 'PM10' in df.columns:
        pm10_lags = [1, 6, 24]
        df = build_lag_features(df, 'PM10', pm10_lags)

    # ========== 6. 气象特征滞后 ==========
    print("   6. 气象特征...")
    weather_cols = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m',
                    'surface_pressure', 'dew_point_2m']

    for col in weather_cols:
        if col in df.columns:
            # 滞后
            df = build_lag_features(df, col, [1, 3, 6])
            # 变化
            df[f'{col}_change_3h'] = df[col] - df[col].shift(3)
            df[f'{col}_change_6h'] = df[col] - df[col].shift(6)

    # ========== 7. 风向分量 (u, v) ==========
    print("   7. 风向分量...")
    if 'wind_direction_10m' in df.columns and 'wind_speed_10m' in df.columns:
        # 转换为弧度
        wind_rad = np.radians(df['wind_direction_10m'])
        df['wind_u'] = df['wind_speed_10m'] * np.sin(wind_rad)  # 东西分量
        df['wind_v'] = df['wind_speed_10m'] * np.cos(wind_rad)  # 南北分量

    # ========== 8. 时间特征增强 ==========
    print("   8. 时间特征增强...")
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 基础时间特征（如果不存在）
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['datetime'].dt.dayofweek
        if 'month' not in df.columns:
            df['month'] = df['datetime'].dt.month
        if 'day_of_year' not in df.columns:
            df['day_of_year'] = df['datetime'].dt.dayofyear

        # 周期编码（如果不存在）
        if 'hour_sin' not in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        if 'day_of_week_sin' not in df.columns:
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        if 'month_sin' not in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # 高峰时段标记
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) |
                             (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)

        # 夜间标记
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    # ========== 9. 代理变量 ==========
    print("   9. 代理变量...")

    # 日温差（逆温层代理）
    if 'temperature_2m' in df.columns:
        df['temp_daily_range'] = df.groupby(df['datetime'].dt.date)['temperature_2m'].transform(
            lambda x: x.max() - x.min()
        )

    # 供暖季标记（11月-3月）
    if 'month' in df.columns:
        df['is_heating_season'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)

    # 春节期间标记（简化：1月下旬-2月中旬）
    if 'month' in df.columns and 'day' in df.columns:
        df['is_spring_festival'] = (
            ((df['month'] == 1) & (df['day'] >= 20)) |
            ((df['month'] == 2) & (df['day'] <= 15))
        ).astype(int)

    # ========== 10. 交互特征 ==========
    print("   10. 交互特征...")

    # 温度 × 湿度（闷热指数）
    if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
        df['temp_humidity_index'] = df['temperature_2m'] * df['relative_humidity_2m'] / 100

    # 风速 × AQI滞后（扩散能力）
    if 'wind_speed_10m' in df.columns and 'AQI_lag_1h' in df.columns:
        df['wind_aqi_interaction'] = df['wind_speed_10m'] * df['AQI_lag_1h']

    # 高峰时段 × AQI滞后
    if 'is_rush_hour' in df.columns and 'AQI_lag_1h' in df.columns:
        df['rush_hour_aqi'] = df['is_rush_hour'] * df['AQI_lag_1h']

    # 低风速标记（不利于扩散）
    if 'wind_speed_10m' in df.columns:
        df['is_low_wind'] = (df['wind_speed_10m'] < 2).astype(int)

    # 高湿度标记（促进二次颗粒物）
    if 'relative_humidity_2m' in df.columns:
        df['is_high_humidity'] = (df['relative_humidity_2m'] > 80).astype(int)

    print(f"   ✅ 特征构建完成，共 {len(df.columns)} 列")

    return df


def select_features(df, target='AQI', correlation_threshold=0.05, max_features=30):
    """
    特征选择（排除其他城市同期AQI）
    """
    print("\n🎯 特征选择...")

    # 排除的列
    exclude_cols = [
        'datetime', 'date', 'AQI', 'year',  # 非特征列
        '南京_AQI', '南通_AQI', '泰州_AQI', '镇江_AQI',  # 其他城市同期AQI
        'holiday_name'  # 文本列
    ]

    # 候选特征
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64']]
    print(f"   候选特征: {len(feature_cols)} 个")

    # 1. 排除缺失率过高的特征 (>20%)
    missing_rate = df[feature_cols].isnull().mean()
    valid_features = missing_rate[missing_rate < 0.2].index.tolist()
    print(f"   缺失率筛选后: {len(valid_features)} 个")

    # 2. 计算与目标变量的相关性
    correlations = df[valid_features].corrwith(df[target]).abs()
    correlations = correlations.dropna().sort_values(ascending=False)

    # 排除相关性过低的特征
    selected = correlations[correlations >= correlation_threshold].index.tolist()
    print(f"   相关性筛选后 (>={correlation_threshold}): {len(selected)} 个")

    # 3. 排除与其他特征高度共线的（保留相关性更高的）
    # 简化：直接取 Top N
    if len(selected) > max_features:
        selected = selected[:max_features]

    print(f"   最终选择: {len(selected)} 个特征")
    print("\n   Top 10 特征 (按与AQI相关性排序):")
    for i, feat in enumerate(selected[:10]):
        print(f"      {i+1}. {feat}: {correlations[feat]:.4f}")

    return selected


def main():
    """主函数"""
    print("=" * 60)
    print("  特征工程 v2 - 不使用其他城市同期AQI")
    print("=" * 60)

    # 读取数据
    input_file = PROCESSED_DATA_DIR / "yangzhou_merged.csv"
    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return

    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"\n📂 读取数据: {len(df)} 条")

    # 构建特征
    df = build_all_features(df)

    # 删除无目标变量的行
    df = df.dropna(subset=['AQI'])
    print(f"   有效数据: {len(df)} 条")

    # 特征选择
    selected_features = select_features(df, max_features=25)

    # 保存完整特征数据
    output_full = FEATURES_DATA_DIR / "yangzhou_features_v2.csv"
    df.to_csv(output_full, index=False)
    print(f"\n💾 完整特征数据已保存: {output_full}")

    # 保存筛选后的特征数据
    output_selected = FEATURES_DATA_DIR / "yangzhou_features_v2_selected.csv"
    cols_to_save = ['datetime', 'AQI'] + selected_features
    cols_to_save = [c for c in cols_to_save if c in df.columns]
    df[cols_to_save].to_csv(output_selected, index=False)
    print(f"💾 筛选特征数据已保存: {output_selected}")

    return df, selected_features


if __name__ == "__main__":
    main()
