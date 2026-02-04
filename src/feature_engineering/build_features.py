# -*- coding: utf-8 -*-
"""
特征工程模块
构建用于 AQI 预测的所有特征
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DATA_DIR, FEATURES_DATA_DIR, DOCS_DIR


# 特征候选池记录
FEATURE_CANDIDATES = []


def record_feature(name, category, description, formula=None):
    """记录特征到候选池"""
    FEATURE_CANDIDATES.append({
        "name": name,
        "category": category,
        "description": description,
        "formula": formula
    })


def add_lag_features(df, columns, lags):
    """
    添加滞后特征

    Args:
        df: DataFrame
        columns: 要添加滞后的列
        lags: 滞后小时数列表

    Returns:
        DataFrame
    """
    print("   添加滞后特征...")

    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        for lag in lags:
            new_col = f"{col}_lag_{lag}h"
            df[new_col] = df[col].shift(lag)
            record_feature(new_col, "滞后特征",
                          f"{col} 的 {lag} 小时滞后值",
                          f"shift({lag})")

    return df


def add_rolling_features(df, columns, windows):
    """
    添加滚动统计特征

    Args:
        df: DataFrame
        columns: 要计算滚动统计的列
        windows: 窗口大小列表（小时）

    Returns:
        DataFrame
    """
    print("   添加滚动统计特征...")

    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        for window in windows:
            # 均值
            mean_col = f"{col}_rolling_mean_{window}h"
            df[mean_col] = df[col].rolling(window=window, min_periods=1).mean()
            record_feature(mean_col, "滚动统计", f"{col} 的 {window} 小时滚动均值")

            # 标准差
            std_col = f"{col}_rolling_std_{window}h"
            df[std_col] = df[col].rolling(window=window, min_periods=1).std()
            record_feature(std_col, "滚动统计", f"{col} 的 {window} 小时滚动标准差")

            # 最大值
            max_col = f"{col}_rolling_max_{window}h"
            df[max_col] = df[col].rolling(window=window, min_periods=1).max()
            record_feature(max_col, "滚动统计", f"{col} 的 {window} 小时滚动最大值")

            # 最小值
            min_col = f"{col}_rolling_min_{window}h"
            df[min_col] = df[col].rolling(window=window, min_periods=1).min()
            record_feature(min_col, "滚动统计", f"{col} 的 {window} 小时滚动最小值")

    return df


def add_change_features(df, columns, periods):
    """
    添加变化率特征

    Args:
        df: DataFrame
        columns: 列名列表
        periods: 时间周期列表（小时）

    Returns:
        DataFrame
    """
    print("   添加变化率特征...")

    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        for period in periods:
            # 绝对变化
            change_col = f"{col}_change_{period}h"
            df[change_col] = df[col] - df[col].shift(period)
            record_feature(change_col, "变化率", f"{col} 的 {period} 小时变化量")

            # 百分比变化
            pct_col = f"{col}_pct_change_{period}h"
            df[pct_col] = df[col].pct_change(periods=period) * 100
            record_feature(pct_col, "变化率", f"{col} 的 {period} 小时变化百分比")

    return df


def add_time_features(df):
    """
    添加时间特征

    Args:
        df: DataFrame

    Returns:
        DataFrame
    """
    print("   添加时间特征...")

    df = df.copy()

    if 'datetime' not in df.columns:
        return df

    # 基础时间特征
    if 'hour' not in df.columns:
        df['hour'] = df['datetime'].dt.hour
        record_feature('hour', "时间特征", "小时 (0-23)")

    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['datetime'].dt.dayofweek
        record_feature('day_of_week', "时间特征", "星期几 (0-6, 周一=0)")

    if 'month' not in df.columns:
        df['month'] = df['datetime'].dt.month
        record_feature('month', "时间特征", "月份 (1-12)")

    if 'day_of_year' not in df.columns:
        df['day_of_year'] = df['datetime'].dt.dayofyear
        record_feature('day_of_year', "时间特征", "一年中的第几天 (1-366)")

    # 周期性编码（正弦/余弦）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    record_feature('hour_sin', "时间特征", "小时正弦编码")
    record_feature('hour_cos', "时间特征", "小时余弦编码")

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    record_feature('month_sin', "时间特征", "月份正弦编码")
    record_feature('month_cos', "时间特征", "月份余弦编码")

    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    record_feature('dow_sin', "时间特征", "星期正弦编码")
    record_feature('dow_cos', "时间特征", "星期余弦编码")

    # 时段特征
    def get_time_period(hour):
        if 6 <= hour < 9:
            return 1  # 早高峰
        elif 9 <= hour < 12:
            return 2  # 上午
        elif 12 <= hour < 14:
            return 3  # 午间
        elif 14 <= hour < 17:
            return 4  # 下午
        elif 17 <= hour < 20:
            return 5  # 晚高峰
        elif 20 <= hour < 23:
            return 6  # 晚间
        else:
            return 7  # 夜间

    df['time_period'] = df['hour'].apply(get_time_period)
    record_feature('time_period', "时间特征", "时段 (1=早高峰,2=上午,3=午间,4=下午,5=晚高峰,6=晚间,7=夜间)")

    # 是否工作时间
    df['is_work_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    record_feature('is_work_hour', "时间特征", "是否工作时间 (8:00-18:00)")

    # 是否通勤时间
    df['is_commute_hour'] = (
        ((df['hour'] >= 7) & (df['hour'] <= 9)) |
        ((df['hour'] >= 17) & (df['hour'] <= 19))
    ).astype(int)
    record_feature('is_commute_hour', "时间特征", "是否通勤时间")

    return df


def add_weather_interaction_features(df):
    """
    添加气象交互特征

    Args:
        df: DataFrame

    Returns:
        DataFrame
    """
    print("   添加气象交互特征...")

    df = df.copy()

    # 温湿度交互
    if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
        # 体感温度（简化公式）
        df['apparent_temp'] = df['temperature_2m'] - 0.55 * (1 - df['relative_humidity_2m']/100) * (df['temperature_2m'] - 14)
        record_feature('apparent_temp', "气象交互", "体感温度")

        # 温湿度乘积
        df['temp_humidity_product'] = df['temperature_2m'] * df['relative_humidity_2m']
        record_feature('temp_humidity_product', "气象交互", "温度×湿度")

    # 风速风向交互
    if 'wind_speed_10m' in df.columns and 'wind_direction_10m' in df.columns:
        # 风向分解为南北/东西分量
        df['wind_u'] = df['wind_speed_10m'] * np.sin(np.radians(df['wind_direction_10m']))
        df['wind_v'] = df['wind_speed_10m'] * np.cos(np.radians(df['wind_direction_10m']))
        record_feature('wind_u', "气象交互", "东西风分量 (正=东风)")
        record_feature('wind_v', "气象交互", "南北风分量 (正=北风)")

    # 气压变化
    if 'surface_pressure' in df.columns:
        df['pressure_change_3h'] = df['surface_pressure'] - df['surface_pressure'].shift(3)
        df['pressure_change_6h'] = df['surface_pressure'] - df['surface_pressure'].shift(6)
        record_feature('pressure_change_3h', "气象交互", "3小时气压变化")
        record_feature('pressure_change_6h', "气象交互", "6小时气压变化")

    # 降水标志
    if 'precipitation' in df.columns:
        df['is_raining'] = (df['precipitation'] > 0.1).astype(int)
        df['rain_intensity'] = pd.cut(df['precipitation'],
                                      bins=[-np.inf, 0, 0.1, 2.5, 8, 16, np.inf],
                                      labels=[0, 1, 2, 3, 4, 5])  # 无/毛毛雨/小雨/中雨/大雨/暴雨
        record_feature('is_raining', "气象交互", "是否降雨")
        record_feature('rain_intensity', "气象交互", "降雨强度等级")

    return df


def add_proxy_features(df):
    """
    添加代理变量（近似不可观测因素）

    Args:
        df: DataFrame

    Returns:
        DataFrame
    """
    print("   添加代理变量...")

    df = df.copy()

    # 逆温层代理：日夜温差
    if 'temperature_2m' in df.columns and 'datetime' in df.columns:
        # 计算当天的最高和最低温度
        df['date'] = df['datetime'].dt.date
        daily_temp = df.groupby('date')['temperature_2m'].agg(['max', 'min'])
        daily_temp['temp_range'] = daily_temp['max'] - daily_temp['min']
        daily_temp = daily_temp.reset_index()

        df = df.merge(daily_temp[['date', 'temp_range']], on='date', how='left')
        df['temp_inversion_proxy'] = df['temp_range']
        df = df.drop(columns=['date', 'temp_range'])
        record_feature('temp_inversion_proxy', "代理变量", "逆温层代理 (日温差)")

    # 秸秆焚烧代理：秋收季节标记
    if 'month' in df.columns:
        df['straw_burning_proxy'] = df['month'].isin([9, 10, 11]).astype(int)
        record_feature('straw_burning_proxy', "代理变量", "秸秆焚烧代理 (秋收季节)")

    # 春节烟花代理
    if 'datetime' in df.columns:
        def is_spring_festival(dt):
            # 简化判断：1-2月的前15天
            if dt.month == 1 and dt.day >= 20:
                return 1
            if dt.month == 2 and dt.day <= 15:
                return 1
            return 0

        df['spring_festival_proxy'] = df['datetime'].apply(is_spring_festival)
        record_feature('spring_festival_proxy', "代理变量", "春节烟花代理")

    # 供暖季代理
    if 'month' in df.columns:
        df['heating_season'] = df['month'].isin([11, 12, 1, 2, 3]).astype(int)
        record_feature('heating_season', "代理变量", "供暖季标记")

    return df


def add_upwind_city_features(df):
    """
    添加上风向城市特征

    Args:
        df: DataFrame

    Returns:
        DataFrame
    """
    print("   添加上风向城市特征...")

    df = df.copy()

    upwind_cities = ['南京', '镇江', '泰州', '南通']

    for city in upwind_cities:
        aqi_col = f"{city}_AQI"
        if aqi_col in df.columns:
            # 滞后特征（污染传输时间）
            for lag in [3, 6, 12]:
                lag_col = f"{city}_aqi_lag_{lag}h"
                df[lag_col] = df[aqi_col].shift(lag)
                record_feature(lag_col, "上风向城市", f"{city} AQI 的 {lag} 小时滞后")

    return df


def build_all_features(df):
    """
    构建所有特征

    Args:
        df: 合并后的原始数据

    Returns:
        DataFrame: 包含所有特征的数据
    """
    global FEATURE_CANDIDATES
    FEATURE_CANDIDATES = []  # 重置

    print("🛠️ 开始构建特征...")

    # 确保按时间排序
    df = df.sort_values('datetime').reset_index(drop=True)

    # 1. 时间特征
    df = add_time_features(df)

    # 2. 滞后特征
    aqi_cols = ['AQI', 'PM2.5', 'PM10']
    df = add_lag_features(df, aqi_cols, lags=[1, 3, 6, 12, 24, 48])

    weather_cols = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'surface_pressure']
    df = add_lag_features(df, weather_cols, lags=[1, 3, 6])

    # 3. 滚动统计特征
    df = add_rolling_features(df, ['AQI', 'PM2.5'], windows=[6, 12, 24, 48])

    # 4. 变化率特征
    df = add_change_features(df, ['AQI', 'PM2.5', 'temperature_2m', 'relative_humidity_2m'], periods=[1, 3, 6, 24])

    # 5. 气象交互特征
    df = add_weather_interaction_features(df)

    # 6. 代理变量
    df = add_proxy_features(df)

    # 7. 上风向城市特征
    df = add_upwind_city_features(df)

    # 删除前面的行（因为滞后特征会产生 NaN）
    initial_rows = len(df)
    df = df.dropna(subset=['AQI'])  # 至少 AQI 不能为空
    final_rows = len(df)
    print(f"   删除缺失行: {initial_rows - final_rows} 行")

    print(f"\n✅ 特征构建完成!")
    print(f"   总特征数: {len(FEATURE_CANDIDATES)}")
    print(f"   数据行数: {len(df)}")

    return df


def save_feature_documentation():
    """保存特征文档"""
    import json

    # 按类别分组
    categories = {}
    for feat in FEATURE_CANDIDATES:
        cat = feat['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(feat)

    # 保存完整候选池
    output_file = DOCS_DIR / "feature_candidates.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_count": len(FEATURE_CANDIDATES),
            "categories": categories,
            "features": FEATURE_CANDIDATES
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📄 特征文档已保存: {output_file}")

    # 打印摘要
    print("\n📊 特征类别统计:")
    for cat, feats in categories.items():
        print(f"   {cat}: {len(feats)} 个")


def main():
    """主函数"""
    print("=" * 60)
    print("  特征工程")
    print("=" * 60)

    # 读取合并后的数据
    input_file = PROCESSED_DATA_DIR / "yangzhou_merged.csv"

    if not input_file.exists():
        print(f"❌ 数据文件不存在: {input_file}")
        print("   请先运行 merge_data.py")
        return

    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    print(f"📖 读取数据: {len(df)} 行, {len(df.columns)} 列")

    # 构建特征
    df_features = build_all_features(df)

    # 保存
    output_file = FEATURES_DATA_DIR / "yangzhou_features.csv"
    df_features.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n✅ 特征数据已保存: {output_file}")
    print(f"   行数: {len(df_features)}")
    print(f"   列数: {len(df_features.columns)}")

    # 保存特征文档
    save_feature_documentation()

    return df_features


if __name__ == "__main__":
    main()
