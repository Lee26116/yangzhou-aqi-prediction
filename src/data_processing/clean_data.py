# -*- coding: utf-8 -*-
"""
数据清洗模块
处理缺失值、异常值、时间对齐等
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def clean_aqi_data(df):
    """
    清洗 AQI 数据

    Args:
        df: 原始 AQI DataFrame

    Returns:
        清洗后的 DataFrame
    """
    print("🧹 清洗 AQI 数据...")

    df = df.copy()

    # 确保 datetime 列是 datetime 类型
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # 设置 datetime 为索引
    if 'datetime' in df.columns:
        df = df.set_index('datetime')

    # 定义数值列
    numeric_cols = ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3',
                    'PM2.5_24h', 'PM10_24h', 'SO2_24h', 'NO2_24h', 'CO_24h', 'O3_24h', 'O3_8h', 'O3_8h_24h']

    # 转换为数值类型
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 异常值处理
    # AQI 范围: 0-500
    if 'AQI' in df.columns:
        df.loc[df['AQI'] < 0, 'AQI'] = np.nan
        df.loc[df['AQI'] > 500, 'AQI'] = np.nan

    # PM2.5 范围: 0-1000
    if 'PM2.5' in df.columns:
        df.loc[df['PM2.5'] < 0, 'PM2.5'] = np.nan
        df.loc[df['PM2.5'] > 1000, 'PM2.5'] = np.nan

    # PM10 范围: 0-1000
    if 'PM10' in df.columns:
        df.loc[df['PM10'] < 0, 'PM10'] = np.nan
        df.loc[df['PM10'] > 1000, 'PM10'] = np.nan

    # 检测并处理异常值 (使用 IQR 方法)
    for col in ['AQI', 'PM2.5', 'PM10']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # 使用 3 倍 IQR，更宽松
            upper_bound = Q3 + 3 * IQR

            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            n_outliers = outliers.sum()
            if n_outliers > 0:
                print(f"   {col}: 发现 {n_outliers} 个异常值，使用插值替换")
                df.loc[outliers, col] = np.nan

    # 处理缺失值 - 使用线性插值
    df = df.interpolate(method='linear', limit=6)  # 最多连续插值 6 小时

    # 前向填充剩余缺失值
    df = df.ffill(limit=3)

    # 后向填充
    df = df.bfill(limit=3)

    # 重置索引
    df = df.reset_index()

    # 统计缺失情况
    print(f"   清洗后缺失值统计:")
    for col in numeric_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"      {col}: {missing} ({missing/len(df)*100:.2f}%)")

    return df


def clean_weather_data(df):
    """
    清洗天气数据

    Args:
        df: 原始天气 DataFrame

    Returns:
        清洗后的 DataFrame
    """
    print("🧹 清洗天气数据...")

    df = df.copy()

    # 确保 datetime 列是 datetime 类型
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # 设置 datetime 为索引
    if 'datetime' in df.columns:
        df = df.set_index('datetime')

    # 定义各变量的合理范围
    ranges = {
        'temperature_2m': (-40, 50),  # 摄氏度
        'relative_humidity_2m': (0, 100),  # 百分比
        'wind_speed_10m': (0, 100),  # m/s
        'wind_direction_10m': (0, 360),  # 度
        'surface_pressure': (900, 1100),  # hPa
        'precipitation': (0, 200),  # mm
        'cloud_cover': (0, 100),  # 百分比
        'boundary_layer_height': (0, 5000),  # 米
        'uv_index': (0, 15),  # UV 指数
    }

    # 处理异常值
    for col, (low, high) in ranges.items():
        if col in df.columns:
            outliers = (df[col] < low) | (df[col] > high)
            n_outliers = outliers.sum()
            if n_outliers > 0:
                print(f"   {col}: 发现 {n_outliers} 个超范围值，设为 NaN")
                df.loc[outliers, col] = np.nan

    # 处理缺失值
    df = df.interpolate(method='linear', limit=6)
    df = df.ffill(limit=3)
    df = df.bfill(limit=3)

    # 重置索引
    df = df.reset_index()

    return df


def align_time_series(dfs, freq='H'):
    """
    对齐多个时间序列数据

    Args:
        dfs: DataFrame 列表
        freq: 频率，H=小时

    Returns:
        对齐后的 DataFrame 列表
    """
    print("⏰ 对齐时间序列...")

    # 找到所有数据的共同时间范围
    min_time = max(df['datetime'].min() for df in dfs if 'datetime' in df.columns)
    max_time = min(df['datetime'].max() for df in dfs if 'datetime' in df.columns)

    print(f"   共同时间范围: {min_time} ~ {max_time}")

    # 创建标准时间索引
    time_index = pd.date_range(start=min_time, end=max_time, freq=freq)

    aligned_dfs = []
    for df in dfs:
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
            df = df.reindex(time_index)
            df = df.interpolate(method='linear', limit=3)
            df = df.reset_index().rename(columns={'index': 'datetime'})
        aligned_dfs.append(df)

    return aligned_dfs


def detect_data_quality_issues(df):
    """
    检测数据质量问题

    Args:
        df: DataFrame

    Returns:
        问题报告 dict
    """
    df = df.copy()

    # 确保 datetime 列是 datetime 类型
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    report = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "missing_values": {},
        "duplicates": 0,
        "time_gaps": [],
    }

    # 缺失值统计
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            report["missing_values"][col] = {
                "count": int(missing),
                "percentage": round(missing / len(df) * 100, 2)
            }

    # 重复行
    if 'datetime' in df.columns:
        report["duplicates"] = int(df.duplicated(subset=['datetime']).sum())

        # 检测时间间隔
        df_sorted = df.sort_values('datetime')
        time_diff = df_sorted['datetime'].diff()
        gaps = time_diff[time_diff > pd.Timedelta(hours=2)]

        for idx in gaps.index[:10]:  # 只记录前 10 个
            report["time_gaps"].append({
                "start": str(df_sorted.loc[idx - 1, 'datetime']),
                "end": str(df_sorted.loc[idx, 'datetime']),
                "duration_hours": time_diff.loc[idx].total_seconds() / 3600
            })

    return report


def main():
    """主函数"""
    print("=" * 60)
    print("  数据清洗")
    print("=" * 60)

    # 读取原始数据
    aqi_file = RAW_DATA_DIR / "yangzhou_aqi_historical.csv"
    weather_file = RAW_DATA_DIR / "yangzhou_weather_historical.csv"

    if not aqi_file.exists():
        print(f"❌ AQI 数据文件不存在: {aqi_file}")
        print("   请先运行 fetch_historical.py 下载数据")
        return

    if not weather_file.exists():
        print(f"❌ 天气数据文件不存在: {weather_file}")
        print("   请先运行 fetch_openmeteo.py 下载数据")
        return

    # 读取数据
    print("\n📖 读取原始数据...")
    aqi_df = pd.read_csv(aqi_file)
    weather_df = pd.read_csv(weather_file)

    print(f"   AQI 数据: {len(aqi_df)} 行")
    print(f"   天气数据: {len(weather_df)} 行")

    # 数据质量检测
    print("\n🔍 检测数据质量问题...")
    aqi_report = detect_data_quality_issues(aqi_df)
    weather_report = detect_data_quality_issues(weather_df)

    # 清洗数据
    aqi_clean = clean_aqi_data(aqi_df)
    weather_clean = clean_weather_data(weather_df)

    # 保存清洗后的数据
    aqi_output = PROCESSED_DATA_DIR / "yangzhou_aqi_cleaned.csv"
    weather_output = PROCESSED_DATA_DIR / "yangzhou_weather_cleaned.csv"

    aqi_clean.to_csv(aqi_output, index=False, encoding='utf-8')
    weather_clean.to_csv(weather_output, index=False, encoding='utf-8')

    print(f"\n✅ 清洗后的数据已保存:")
    print(f"   AQI: {aqi_output}")
    print(f"   天气: {weather_output}")

    # 保存数据质量报告
    import json
    report_file = PROCESSED_DATA_DIR / "data_quality_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            "aqi": aqi_report,
            "weather": weather_report
        }, f, ensure_ascii=False, indent=2)

    print(f"   数据质量报告: {report_file}")


if __name__ == "__main__":
    main()
