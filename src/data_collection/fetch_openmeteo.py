# -*- coding: utf-8 -*-
"""
从 Open-Meteo 下载历史天气数据
免费 API，无需 Key
"""

import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    RAW_DATA_DIR, DATA_COLLECTION_CONFIG, YANGZHOU_CONFIG
)

# API URL
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# 天气参数
WEATHER_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "cloud_cover",
    "weather_code",
    "dew_point_2m",
    "apparent_temperature",
    "rain",
    "snowfall",
    "visibility",
    "wind_gusts_10m",
    "soil_temperature_0cm",
    "boundary_layer_height",
    "uv_index"
]


def fetch_weather_for_location(lat, lon, start_date, end_date, location_name="扬州"):
    """
    获取指定位置的历史天气数据

    Args:
        lat: 纬度
        lon: 经度
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        location_name: 位置名称

    Returns:
        DataFrame
    """
    print(f"\n📥 下载 {location_name} 历史天气数据...")
    print(f"   坐标: ({lat}, {lon})")
    print(f"   时间范围: {start_date} ~ {end_date}")

    # Open-Meteo 单次请求最多支持 366 天，需要分批下载
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []

    current_start = start
    while current_start < end:
        # 每批最多 365 天
        current_end = min(current_start + timedelta(days=365), end)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": current_start.strftime("%Y-%m-%d"),
            "end_date": current_end.strftime("%Y-%m-%d"),
            "hourly": ",".join(WEATHER_PARAMS),
            "timezone": "Asia/Shanghai"
        }

        try:
            response = requests.get(ARCHIVE_API_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            if "hourly" in data:
                hourly = data["hourly"]
                df = pd.DataFrame(hourly)
                all_data.append(df)
                print(f"   ✅ {current_start.strftime('%Y-%m-%d')} ~ {current_end.strftime('%Y-%m-%d')}: {len(df)} 条记录")

        except Exception as e:
            print(f"   ❌ {current_start.strftime('%Y-%m-%d')} ~ {current_end.strftime('%Y-%m-%d')}: {e}")

        current_start = current_end + timedelta(days=1)

    if not all_data:
        print("   ❌ 没有获取到数据")
        return None

    # 合并数据
    combined_df = pd.concat(all_data, ignore_index=True)

    # 转换时间列
    combined_df['datetime'] = pd.to_datetime(combined_df['time'])
    combined_df = combined_df.drop(columns=['time'])

    # 去重（可能有重叠）
    combined_df = combined_df.drop_duplicates(subset=['datetime']).reset_index(drop=True)

    # 排序
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)

    return combined_df


def download_yangzhou_weather():
    """下载扬州历史天气数据"""
    start_date = DATA_COLLECTION_CONFIG["historical_start_date"]
    end_date = DATA_COLLECTION_CONFIG["historical_end_date"]

    df = fetch_weather_for_location(
        lat=YANGZHOU_CONFIG["latitude"],
        lon=YANGZHOU_CONFIG["longitude"],
        start_date=start_date,
        end_date=end_date,
        location_name="扬州"
    )

    if df is not None:
        output_file = RAW_DATA_DIR / "yangzhou_weather_historical.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✅ 扬州历史天气数据已保存: {output_file}")
        print(f"   记录数: {len(df)}")
        print(f"   时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
        print(f"   列: {list(df.columns)}")

    return df


def download_upwind_cities_weather():
    """下载上风向城市的历史天气数据"""
    start_date = DATA_COLLECTION_CONFIG["historical_start_date"]
    end_date = DATA_COLLECTION_CONFIG["historical_end_date"]

    all_city_data = {}

    for city_name, coords in YANGZHOU_CONFIG["upwind_cities"].items():
        df = fetch_weather_for_location(
            lat=coords["lat"],
            lon=coords["lon"],
            start_date=start_date,
            end_date=end_date,
            location_name=city_name
        )

        if df is not None:
            # 重命名列，添加城市前缀
            rename_dict = {col: f"{city_name}_{col}" for col in df.columns if col != 'datetime'}
            df = df.rename(columns=rename_dict)
            all_city_data[city_name] = df

    if not all_city_data:
        print("❌ 没有获取到上风向城市天气数据")
        return None

    # 合并所有城市数据
    merged_df = None
    for city_name, df in all_city_data.items():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = merged_df.merge(df, on='datetime', how='outer')

    # 排序
    merged_df = merged_df.sort_values('datetime').reset_index(drop=True)

    # 保存
    output_file = RAW_DATA_DIR / "upwind_cities_weather_historical.csv"
    merged_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n✅ 上风向城市历史天气数据已保存: {output_file}")
    print(f"   记录数: {len(merged_df)}")

    return merged_df


def main():
    """主函数"""
    print("=" * 60)
    print("  Open-Meteo 历史天气数据下载")
    print("=" * 60)

    # 1. 下载扬州天气数据
    download_yangzhou_weather()

    # 2. 下载上风向城市天气数据
    download_upwind_cities_weather()


if __name__ == "__main__":
    main()
