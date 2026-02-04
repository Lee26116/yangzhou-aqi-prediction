# -*- coding: utf-8 -*-
"""
从 WAQI 获取实时空气质量数据
"""

import sys
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import RAW_DATA_DIR, WAQI_TOKEN, YANGZHOU_CONFIG


def fetch_realtime_aqi(city="yangzhou"):
    """
    获取实时 AQI 数据

    Args:
        city: 城市名或站点 ID

    Returns:
        dict: AQI 数据
    """
    url = f"https://api.waqi.info/feed/{city}/?token={WAQI_TOKEN}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "ok":
            return data["data"]
        else:
            print(f"API 返回错误: {data}")
            return None

    except Exception as e:
        print(f"请求失败: {e}")
        return None


def parse_waqi_data(data):
    """
    解析 WAQI 返回的数据

    Args:
        data: WAQI API 返回的 data 部分

    Returns:
        dict: 解析后的数据
    """
    if not data:
        return None

    result = {
        "fetch_time": datetime.now().isoformat(),
        "aqi": data.get("aqi"),
        "dominant_pollutant": data.get("dominentpol"),
        "city_name": data.get("city", {}).get("name"),
        "station_idx": data.get("idx"),
    }

    # 解析各项指标
    iaqi = data.get("iaqi", {})
    if isinstance(iaqi, dict):
        for key, value in iaqi.items():
            if isinstance(value, dict):
                result[key] = value.get("v")
            else:
                result[key] = value

    # 解析时间
    time_info = data.get("time", {})
    result["data_time"] = time_info.get("iso")

    # 解析预报
    forecast = data.get("forecast", {}).get("daily", {})
    if forecast:
        # 提取 PM2.5 预报
        pm25_forecast = forecast.get("pm25", [])
        result["forecast_pm25"] = pm25_forecast

        # 提取 PM10 预报
        pm10_forecast = forecast.get("pm10", [])
        result["forecast_pm10"] = pm10_forecast

    return result


def get_yangzhou_realtime():
    """获取扬州实时 AQI 数据"""
    print("📡 获取扬州实时 AQI 数据...")

    data = fetch_realtime_aqi("yangzhou")
    parsed = parse_waqi_data(data)

    if parsed:
        print(f"   ✅ AQI: {parsed.get('aqi')}")
        print(f"   主要污染物: {parsed.get('dominant_pollutant')}")
        print(f"   PM2.5: {parsed.get('pm25')}")
        print(f"   PM10: {parsed.get('pm10')}")
        print(f"   数据时间: {parsed.get('data_time')}")

    return parsed


def get_upwind_cities_realtime():
    """获取上风向城市实时 AQI 数据"""
    print("\n📡 获取上风向城市实时 AQI 数据...")

    cities = {
        "南京": "nanjing",
        "镇江": "zhenjiang",
        "泰州": "taizhou",
        "南通": "nantong"
    }

    results = {}

    for cn_name, en_name in cities.items():
        data = fetch_realtime_aqi(en_name)
        parsed = parse_waqi_data(data)
        if parsed:
            results[cn_name] = parsed
            print(f"   ✅ {cn_name}: AQI={parsed.get('aqi')}, PM2.5={parsed.get('pm25')}")
        else:
            print(f"   ❌ {cn_name}: 获取失败")

    return results


def save_realtime_data(yangzhou_data, upwind_data):
    """保存实时数据到文件（追加模式）"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存扬州数据
    if yangzhou_data:
        file_path = RAW_DATA_DIR / "realtime" / "yangzhou_realtime.jsonl"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(yangzhou_data, ensure_ascii=False) + '\n')

    # 保存上风向城市数据
    if upwind_data:
        file_path = RAW_DATA_DIR / "realtime" / "upwind_cities_realtime.jsonl"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        record = {
            "fetch_time": datetime.now().isoformat(),
            "cities": upwind_data
        }
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    """主函数"""
    print("=" * 60)
    print("  WAQI 实时 AQI 数据采集")
    print("=" * 60)

    # 获取扬州数据
    yangzhou = get_yangzhou_realtime()

    # 获取上风向城市数据
    upwind = get_upwind_cities_realtime()

    # 保存数据
    save_realtime_data(yangzhou, upwind)

    print("\n✅ 数据采集完成")

    return yangzhou, upwind


if __name__ == "__main__":
    main()
