# -*- coding: utf-8 -*-
"""
从 quotsoft.net 下载历史空气质量数据
数据格式: china_cities_YYYYMMDD.csv
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import RAW_DATA_DIR, DATA_COLLECTION_CONFIG, YANGZHOU_CONFIG

# 历史 AQI 数据目录
HISTORICAL_AQI_DIR = RAW_DATA_DIR / "historical_aqi"

# quotsoft.net 基础 URL
BASE_URL = "https://quotsoft.net/air/data/china_cities_{date}.csv"


def download_single_day(date_str):
    """
    下载单日数据

    Args:
        date_str: 日期字符串，格式 YYYYMMDD

    Returns:
        (date_str, success, message)
    """
    url = BASE_URL.format(date=date_str)
    output_file = HISTORICAL_AQI_DIR / f"china_cities_{date_str}.csv"

    # 如果文件已存在且大小合理，跳过
    if output_file.exists() and output_file.stat().st_size > 1000:
        return (date_str, True, "已存在")

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            # 保存原始数据
            with open(output_file, 'wb') as f:
                f.write(response.content)
            return (date_str, True, "下载成功")
        else:
            return (date_str, False, f"HTTP {response.status_code}")
    except Exception as e:
        return (date_str, False, str(e))


def download_date_range(start_date, end_date, max_workers=5):
    """
    下载日期范围内的所有数据

    Args:
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        max_workers: 并发下载数
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # 生成日期列表
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)

    print(f"📥 准备下载 {len(dates)} 天的数据...")
    print(f"   时间范围: {start_date} ~ {end_date}")

    success_count = 0
    fail_count = 0
    skip_count = 0

    # 使用线程池并发下载
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_single_day, d): d for d in dates}

        for i, future in enumerate(as_completed(futures)):
            date_str, success, message = future.result()
            if success:
                if message == "已存在":
                    skip_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
                print(f"   ❌ {date_str}: {message}")

            # 每 100 个显示进度
            if (i + 1) % 100 == 0:
                print(f"   进度: {i+1}/{len(dates)}")

    print(f"\n✅ 下载完成!")
    print(f"   成功: {success_count}, 跳过: {skip_count}, 失败: {fail_count}")


def extract_yangzhou_data():
    """
    从所有日文件中提取扬州数据，合并成单个文件
    """
    print("\n📊 提取扬州数据...")

    all_files = sorted(HISTORICAL_AQI_DIR.glob("china_cities_*.csv"))
    print(f"   找到 {len(all_files)} 个数据文件")

    all_data = []

    for i, file_path in enumerate(all_files):
        try:
            df = pd.read_csv(file_path, encoding='utf-8')

            # 检查扬州是否在列中
            if '扬州' not in df.columns:
                continue

            # 提取扬州数据
            yangzhou_df = df[['date', 'hour', 'type', '扬州']].copy()
            yangzhou_df = yangzhou_df.rename(columns={'扬州': 'value'})

            all_data.append(yangzhou_df)

        except Exception as e:
            print(f"   ⚠️ 读取失败: {file_path.name} - {e}")

        if (i + 1) % 100 == 0:
            print(f"   进度: {i+1}/{len(all_files)}")

    if not all_data:
        print("   ❌ 没有找到扬州数据")
        return None

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)

    # 转换为宽格式（每个污染物一列）
    pivot_df = combined_df.pivot_table(
        index=['date', 'hour'],
        columns='type',
        values='value',
        aggfunc='first'
    ).reset_index()

    # 重命名列
    pivot_df.columns.name = None

    # 创建 datetime 列
    pivot_df['datetime'] = pd.to_datetime(
        pivot_df['date'].astype(str) + ' ' + pivot_df['hour'].astype(str).str.zfill(2) + ':00:00'
    )

    # 排序
    pivot_df = pivot_df.sort_values('datetime').reset_index(drop=True)

    # 保存
    output_file = RAW_DATA_DIR / "yangzhou_aqi_historical.csv"
    pivot_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n✅ 扬州历史 AQI 数据已保存: {output_file}")
    print(f"   记录数: {len(pivot_df)}")
    print(f"   时间范围: {pivot_df['datetime'].min()} ~ {pivot_df['datetime'].max()}")
    print(f"   列: {list(pivot_df.columns)}")

    return pivot_df


def extract_upwind_cities_data():
    """
    提取上风向城市的 AQI 数据（用于跨区域污染传输特征）
    """
    print("\n📊 提取上风向城市数据...")

    upwind_cities = list(YANGZHOU_CONFIG['upwind_cities'].keys())
    print(f"   上风向城市: {upwind_cities}")

    all_files = sorted(HISTORICAL_AQI_DIR.glob("china_cities_*.csv"))

    all_data = []

    for i, file_path in enumerate(all_files):
        try:
            df = pd.read_csv(file_path, encoding='utf-8')

            # 检查城市是否在列中
            available_cities = [c for c in upwind_cities if c in df.columns]
            if not available_cities:
                continue

            # 提取数据
            cols = ['date', 'hour', 'type'] + available_cities
            city_df = df[cols].copy()

            all_data.append(city_df)

        except Exception as e:
            pass

        if (i + 1) % 100 == 0:
            print(f"   进度: {i+1}/{len(all_files)}")

    if not all_data:
        print("   ❌ 没有找到上风向城市数据")
        return None

    # 合并
    combined_df = pd.concat(all_data, ignore_index=True)

    # 保存宽格式数据（每个城市的每个指标一列）
    result_dfs = []

    for city in upwind_cities:
        if city in combined_df.columns:
            city_data = combined_df[['date', 'hour', 'type', city]].copy()
            city_pivot = city_data.pivot_table(
                index=['date', 'hour'],
                columns='type',
                values=city,
                aggfunc='first'
            ).reset_index()
            city_pivot.columns.name = None

            # 重命名列，添加城市前缀
            rename_dict = {col: f"{city}_{col}" for col in city_pivot.columns
                           if col not in ['date', 'hour']}
            city_pivot = city_pivot.rename(columns=rename_dict)

            result_dfs.append(city_pivot)

    if not result_dfs:
        return None

    # 合并所有城市数据
    merged_df = result_dfs[0]
    for df in result_dfs[1:]:
        merged_df = merged_df.merge(df, on=['date', 'hour'], how='outer')

    # 创建 datetime 列
    merged_df['datetime'] = pd.to_datetime(
        merged_df['date'].astype(str) + ' ' + merged_df['hour'].astype(str).str.zfill(2) + ':00:00'
    )

    # 排序
    merged_df = merged_df.sort_values('datetime').reset_index(drop=True)

    # 保存
    output_file = RAW_DATA_DIR / "upwind_cities_aqi_historical.csv"
    merged_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n✅ 上风向城市历史 AQI 数据已保存: {output_file}")
    print(f"   记录数: {len(merged_df)}")

    return merged_df


def main():
    """主函数"""
    print("=" * 60)
    print("  扬州空气质量历史数据下载")
    print("=" * 60)

    # 获取配置
    start_date = DATA_COLLECTION_CONFIG["historical_start_date"]
    end_date = DATA_COLLECTION_CONFIG["historical_end_date"]

    # 1. 下载历史数据
    download_date_range(start_date, end_date, max_workers=10)

    # 2. 提取扬州数据
    extract_yangzhou_data()

    # 3. 提取上风向城市数据
    extract_upwind_cities_data()


if __name__ == "__main__":
    main()
