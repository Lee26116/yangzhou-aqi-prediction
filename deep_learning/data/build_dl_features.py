# -*- coding: utf-8 -*-
"""
深度学习专用特征工程
构建约51个特征：扬州AQI/污染物、气象、时间编码、上风向城市空间滞后、区域统计、风向感知
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DATA_DIR, YANGZHOU_CONFIG
from deep_learning.dl_config import (
    DL_FEATURES_DIR, SPATIAL_LAG_CONFIG,
    CITY_DISTANCES, CITY_BEARINGS
)


def load_merged_data():
    """加载合并后的数据"""
    merged_file = PROCESSED_DATA_DIR / "yangzhou_merged.csv"
    if not merged_file.exists():
        raise FileNotFoundError(f"合并数据不存在: {merged_file}")

    df = pd.read_csv(merged_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def build_yangzhou_pollution_features(df):
    """扬州AQI和污染物特征"""
    features = pd.DataFrame(index=df.index)

    pollution_cols = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO', 'SO2']
    for col in pollution_cols:
        if col in df.columns:
            features[col] = df[col]
        else:
            features[col] = np.nan

    return features


def build_weather_features(df):
    """扬州气象特征"""
    features = pd.DataFrame(index=df.index)

    weather_mapping = {
        'temperature_2m': 'temperature_2m',
        'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed',
        'wind_direction_10m': 'wind_direction',
        'surface_pressure': 'pressure',
        'precipitation': 'precipitation',
        'cloud_cover': 'cloud_cover',
        'dew_point_2m': 'dew_point',
        'boundary_layer_height': 'boundary_layer_height',
        'uv_index': 'uv_index',
        'wind_gusts_10m': 'wind_gusts',
    }

    # 尝试直接列名和可能的别名
    for orig_col, feat_name in weather_mapping.items():
        if orig_col in df.columns:
            features[feat_name] = df[orig_col]
        elif feat_name in df.columns:
            features[feat_name] = df[feat_name]
        else:
            features[feat_name] = np.nan

    # 确保 humidity 列存在（可能来自不同名称）
    if features['humidity'].isna().all() and 'humidity' in df.columns:
        features['humidity'] = df['humidity']

    return features


def build_time_features(df):
    """时间编码特征"""
    features = pd.DataFrame(index=df.index)

    dt = df['datetime']
    hour = dt.dt.hour
    month = dt.dt.month
    dow = dt.dt.dayofweek

    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # 二值特征
    if 'is_weekend' in df.columns:
        features['is_weekend'] = df['is_weekend'].astype(float)
    else:
        features['is_weekend'] = (dow >= 5).astype(float)

    if 'is_holiday' in df.columns:
        features['is_holiday'] = df['is_holiday'].astype(float)
    else:
        features['is_holiday'] = 0.0

    # 供暖季 (11月15日 - 3月15日)
    features['is_heating_season'] = ((month >= 11) | (month <= 3)).astype(float)

    return features


def build_upwind_aqi_features(df):
    """上风向城市原始AQI特征"""
    features = pd.DataFrame(index=df.index)

    cities = ['南京', '镇江', '泰州', '南通']
    for city in cities:
        col = f'{city}_AQI'
        if col in df.columns:
            features[f'{city}_AQI'] = df[col]
        else:
            features[f'{city}_AQI'] = np.nan

    return features


def build_spatial_lag_features(df):
    """空间滞后特征（上风向城市AQI/PM2.5的时间滞后）"""
    features = pd.DataFrame(index=df.index)

    for city, config in SPATIAL_LAG_CONFIG.items():
        # AQI 滞后
        aqi_col = f'{city}_AQI'
        if aqi_col in df.columns:
            for lag in config['lags']:
                features[f'{city}_AQI_lag_{lag}h'] = df[aqi_col].shift(lag)

        # PM2.5 滞后
        pm25_col = f'{city}_PM2.5'
        if pm25_col in df.columns:
            for lag in config.get('pm25_lags', []):
                features[f'{city}_PM25_lag_{lag}h'] = df[pm25_col].shift(lag)

    return features


def build_regional_stats(df):
    """区域统计特征"""
    features = pd.DataFrame(index=df.index)

    cities = ['南京', '镇江', '泰州', '南通']
    city_aqi_cols = [f'{c}_AQI' for c in cities if f'{c}_AQI' in df.columns]

    if city_aqi_cols:
        city_aqi = df[city_aqi_cols]
        features['regional_aqi_mean'] = city_aqi.mean(axis=1)
        features['regional_aqi_std'] = city_aqi.std(axis=1)
        features['regional_aqi_max'] = city_aqi.max(axis=1)
    else:
        features['regional_aqi_mean'] = np.nan
        features['regional_aqi_std'] = np.nan
        features['regional_aqi_max'] = np.nan

    return features


def build_wind_aware_features(df):
    """风向感知特征：根据当前风向加权上风向城市的AQI/PM2.5"""
    features = pd.DataFrame(index=df.index)

    # 获取风向列
    wind_dir_col = None
    for col_name in ['wind_direction_10m', 'wind_direction']:
        if col_name in df.columns:
            wind_dir_col = col_name
            break

    if wind_dir_col is None:
        features['upwind_aqi'] = np.nan
        features['upwind_pm25'] = np.nan
        return features

    wind_dir = df[wind_dir_col].values

    # 计算每个城市的风向权重
    upwind_aqi = np.zeros(len(df))
    upwind_pm25 = np.zeros(len(df))
    total_weight = np.zeros(len(df))

    for city, bearing in CITY_BEARINGS.items():
        aqi_col = f'{city}_AQI'
        pm25_col = f'{city}_PM2.5'
        distance = CITY_DISTANCES[city]

        if aqi_col not in df.columns:
            continue

        # 风从城市方向吹来的权重
        # 如果风向与城市方位角相差<90度，该城市对扬州有影响
        angle_diff = np.abs(wind_dir - bearing)
        angle_diff = np.minimum(angle_diff, 360 - angle_diff)

        # 余弦权重（角度差越小权重越大），仅当角度差<90度时
        weight = np.cos(np.radians(angle_diff))
        weight = np.maximum(weight, 0)  # 负权重置零

        # 距离衰减
        weight *= 1.0 / (distance / 30.0)

        city_aqi = df[aqi_col].fillna(0).values
        upwind_aqi += weight * city_aqi
        total_weight += weight

        if pm25_col in df.columns:
            city_pm25 = df[pm25_col].fillna(0).values
            upwind_pm25 += weight * city_pm25

    # 归一化
    mask = total_weight > 0
    upwind_aqi[mask] /= total_weight[mask]
    upwind_pm25[mask] /= total_weight[mask]
    upwind_aqi[~mask] = np.nan
    upwind_pm25[~mask] = np.nan

    features['upwind_aqi'] = upwind_aqi
    features['upwind_pm25'] = upwind_pm25

    return features


def build_all_features():
    """构建所有DL特征"""
    print("=" * 60)
    print("  深度学习特征工程")
    print("=" * 60)

    # 加载数据
    print("\n📖 加载合并数据...")
    df = load_merged_data()
    print(f"   数据形状: {df.shape}")
    print(f"   时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"   列名: {list(df.columns)}")

    # 构建各组特征
    print("\n🔧 构建特征...")

    print("   [1/7] 扬州污染物特征...")
    pollution = build_yangzhou_pollution_features(df)
    print(f"         {pollution.shape[1]} 个特征")

    print("   [2/7] 气象特征...")
    weather = build_weather_features(df)
    print(f"         {weather.shape[1]} 个特征")

    print("   [3/7] 时间编码特征...")
    time_feats = build_time_features(df)
    print(f"         {time_feats.shape[1]} 个特征")

    print("   [4/7] 上风向城市原始AQI...")
    upwind_aqi = build_upwind_aqi_features(df)
    print(f"         {upwind_aqi.shape[1]} 个特征")

    print("   [5/7] 空间滞后特征...")
    spatial_lag = build_spatial_lag_features(df)
    print(f"         {spatial_lag.shape[1]} 个特征")

    print("   [6/7] 区域统计特征...")
    regional = build_regional_stats(df)
    print(f"         {regional.shape[1]} 个特征")

    print("   [7/7] 风向感知特征...")
    wind_aware = build_wind_aware_features(df)
    print(f"         {wind_aware.shape[1]} 个特征")

    # 合并所有特征
    all_features = pd.concat([
        df[['datetime']],
        pollution,
        weather,
        time_feats,
        upwind_aqi,
        spatial_lag,
        regional,
        wind_aware,
    ], axis=1)

    # 去除全为NaN的列
    na_cols = all_features.columns[all_features.isna().all()]
    if len(na_cols) > 0:
        print(f"\n   ⚠️ 移除全空列: {list(na_cols)}")
        all_features = all_features.drop(columns=na_cols)

    # 去除空间滞后导致的头部NaN行
    max_lag = 12  # 最大滞后12小时
    all_features = all_features.iloc[max_lag:].reset_index(drop=True)

    # 填充剩余NaN（前向+后向）
    numeric_cols = all_features.select_dtypes(include=[np.number]).columns
    all_features[numeric_cols] = all_features[numeric_cols].ffill().bfill()

    total_features = len(all_features.columns) - 1  # 减去 datetime
    print(f"\n📊 特征汇总:")
    print(f"   总特征数: {total_features}")
    print(f"   总样本数: {len(all_features)}")
    print(f"   时间范围: {all_features['datetime'].min()} ~ {all_features['datetime'].max()}")

    # 保存
    output_file = DL_FEATURES_DIR / "dl_features.csv"
    all_features.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ DL 特征已保存: {output_file}")

    # 保存特征名列表
    feature_names = [c for c in all_features.columns if c != 'datetime']
    names_file = DL_FEATURES_DIR / "feature_names.txt"
    with open(names_file, 'w') as f:
        f.write('\n'.join(feature_names))
    print(f"   特征名列表: {names_file}")

    return all_features


def main():
    build_all_features()


if __name__ == "__main__":
    main()
