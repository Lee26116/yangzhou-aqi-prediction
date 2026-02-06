# -*- coding: utf-8 -*-
"""
多源数据合并模块
将 AQI、天气、节假日等数据合并
"""

import sys
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_cleaned_data():
    """
    加载清洗后的数据

    Returns:
        tuple: (aqi_df, weather_df)
    """
    aqi_file = PROCESSED_DATA_DIR / "yangzhou_aqi_cleaned.csv"
    weather_file = PROCESSED_DATA_DIR / "yangzhou_weather_cleaned.csv"

    aqi_df = pd.read_csv(aqi_file)
    weather_df = pd.read_csv(weather_file)

    # 转换 datetime
    aqi_df['datetime'] = pd.to_datetime(aqi_df['datetime'])
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

    return aqi_df, weather_df


def load_calendar_data():
    """
    加载日历数据

    Returns:
        DataFrame
    """
    calendar_file = RAW_DATA_DIR / "china_calendar.csv"

    if calendar_file.exists():
        df = pd.read_csv(calendar_file)
        df['date'] = pd.to_datetime(df['date'])
        return df

    return None


def load_upwind_cities_data():
    """
    加载上风向城市数据

    Returns:
        tuple: (aqi_df, weather_df) or (None, None)
    """
    aqi_file = RAW_DATA_DIR / "upwind_cities_aqi_historical.csv"
    weather_file = RAW_DATA_DIR / "upwind_cities_weather_historical.csv"

    aqi_df = None
    weather_df = None

    if aqi_file.exists():
        aqi_df = pd.read_csv(aqi_file)
        aqi_df['datetime'] = pd.to_datetime(aqi_df['datetime'])

    if weather_file.exists():
        weather_df = pd.read_csv(weather_file)
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

    return aqi_df, weather_df


def merge_all_data():
    """
    合并所有数据源

    Returns:
        DataFrame: 合并后的数据
    """
    print("🔀 合并多源数据...")

    # 加载主数据
    aqi_df, weather_df = load_cleaned_data()
    print(f"   AQI 数据: {len(aqi_df)} 行")
    print(f"   天气数据: {len(weather_df)} 行")

    # 合并 AQI 和天气数据
    merged_df = aqi_df.merge(weather_df, on='datetime', how='outer')
    print(f"   合并后: {len(merged_df)} 行")

    # 加载日历数据
    calendar_df = load_calendar_data()
    if calendar_df is not None:
        # 提取日期用于合并
        merged_df['date'] = merged_df['datetime'].dt.date
        merged_df['date'] = pd.to_datetime(merged_df['date'])

        calendar_cols = ['date', 'day_of_week', 'is_weekend', 'is_holiday',
                        'is_workday', 'season', 'is_harvest_season', 'holiday_name']
        calendar_cols = [c for c in calendar_cols if c in calendar_df.columns]

        merged_df = merged_df.merge(calendar_df[calendar_cols], on='date', how='left')
        print(f"   添加日历数据后: {len(merged_df)} 行")

    # 加载上风向城市数据
    upwind_aqi, upwind_weather = load_upwind_cities_data()

    if upwind_aqi is not None:
        # 保留 AQI、PM2.5、PM10 列
        aqi_cols = [c for c in upwind_aqi.columns if 'AQI' in c or 'PM2.5' in c or 'PM10' in c or c == 'datetime']
        if aqi_cols:
            merged_df = merged_df.merge(upwind_aqi[aqi_cols], on='datetime', how='left')
            print(f"   添加上风向城市 AQI 后: {len(merged_df)} 行")

    # 排序
    merged_df = merged_df.sort_values('datetime').reset_index(drop=True)

    # 删除 date 和 hour 列（如果有的话），保留 datetime
    cols_to_drop = []
    if 'date' in merged_df.columns and 'datetime' in merged_df.columns:
        cols_to_drop.append('date')
    if 'hour' in merged_df.columns:
        cols_to_drop.append('hour')

    if cols_to_drop:
        merged_df = merged_df.drop(columns=cols_to_drop)

    return merged_df


def add_time_features(df):
    """
    添加时间特征

    Args:
        df: 带 datetime 列的 DataFrame

    Returns:
        添加了时间特征的 DataFrame
    """
    print("⏰ 添加时间特征...")

    df = df.copy()

    if 'datetime' not in df.columns:
        return df

    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_year'] = df['datetime'].dt.dayofyear
    df['week_of_year'] = df['datetime'].dt.isocalendar().week

    # 周期性编码
    import numpy as np
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # 如果没有 is_weekend 列，添加
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['day_of_week'] >= 5

    # 如果没有 season 列，添加
    if 'season' not in df.columns:
        df['season'] = df['month'].apply(lambda m: 1 if m in [3,4,5] else (2 if m in [6,7,8] else (3 if m in [9,10,11] else 4)))

    return df


def main():
    """主函数"""
    print("=" * 60)
    print("  多源数据合并")
    print("=" * 60)

    # 检查必要文件
    required_files = [
        PROCESSED_DATA_DIR / "yangzhou_aqi_cleaned.csv",
        PROCESSED_DATA_DIR / "yangzhou_weather_cleaned.csv"
    ]

    for f in required_files:
        if not f.exists():
            print(f"❌ 必要文件不存在: {f}")
            print("   请先运行 clean_data.py")
            return

    # 合并数据
    merged_df = merge_all_data()

    # 添加时间特征
    merged_df = add_time_features(merged_df)

    # 保存
    output_file = PROCESSED_DATA_DIR / "yangzhou_merged.csv"
    merged_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n✅ 合并后的数据已保存: {output_file}")
    print(f"   总记录数: {len(merged_df)}")
    print(f"   时间范围: {merged_df['datetime'].min()} ~ {merged_df['datetime'].max()}")
    print(f"   列数: {len(merged_df.columns)}")
    print(f"   列名: {list(merged_df.columns)}")

    # 数据摘要
    print("\n📊 数据摘要:")
    if 'AQI' in merged_df.columns:
        print(f"   AQI 均值: {merged_df['AQI'].mean():.1f}")
        print(f"   AQI 范围: {merged_df['AQI'].min():.0f} ~ {merged_df['AQI'].max():.0f}")

    return merged_df


if __name__ == "__main__":
    main()
