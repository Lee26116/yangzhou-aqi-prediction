# -*- coding: utf-8 -*-
"""
从高德地图获取实时天气和交通数据
"""

import sys
import requests
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import RAW_DATA_DIR, AMAP_KEY, YANGZHOU_CONFIG


def fetch_weather(city_code="321000"):
    """
    获取实时天气数据

    Args:
        city_code: 城市行政区划代码

    Returns:
        dict: 天气数据
    """
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "key": AMAP_KEY,
        "city": city_code,
        "extensions": "all"  # all 返回预报，base 返回实时
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "1":
            return data
        else:
            print(f"API 返回错误: {data}")
            return None

    except Exception as e:
        print(f"请求失败: {e}")
        return None


def fetch_traffic(location, radius=5000):
    """
    获取交通态势数据

    Args:
        location: 中心点坐标 "经度,纬度"
        radius: 半径（米）

    Returns:
        dict: 交通数据
    """
    url = "https://restapi.amap.com/v3/traffic/status/circle"
    params = {
        "key": AMAP_KEY,
        "location": location,
        "radius": radius,
        "extensions": "all"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "1":
            return data
        else:
            print(f"API 返回错误: {data}")
            return None

    except Exception as e:
        print(f"请求失败: {e}")
        return None


def parse_weather_data(data):
    """
    解析高德天气数据

    Args:
        data: API 返回的原始数据

    Returns:
        dict: 解析后的数据
    """
    if not data:
        return None

    result = {
        "fetch_time": datetime.now().isoformat(),
        "city": data.get("forecasts", [{}])[0].get("city"),
        "province": data.get("forecasts", [{}])[0].get("province"),
    }

    forecasts = data.get("forecasts", [])
    if forecasts:
        forecast = forecasts[0]
        result["report_time"] = forecast.get("reporttime")

        # 解析每日预报
        casts = forecast.get("casts", [])
        daily_forecasts = []
        for cast in casts:
            daily_forecasts.append({
                "date": cast.get("date"),
                "week": cast.get("week"),
                "dayweather": cast.get("dayweather"),
                "nightweather": cast.get("nightweather"),
                "daytemp": cast.get("daytemp"),
                "nighttemp": cast.get("nighttemp"),
                "daywind": cast.get("daywind"),
                "nightwind": cast.get("nightwind"),
                "daypower": cast.get("daypower"),
                "nightpower": cast.get("nightpower"),
            })
        result["forecasts"] = daily_forecasts

    return result


def parse_traffic_data(data):
    """
    解析高德交通数据

    Args:
        data: API 返回的原始数据

    Returns:
        dict: 解析后的数据
    """
    if not data:
        return None

    trafficinfo = data.get("trafficinfo", {})

    result = {
        "fetch_time": datetime.now().isoformat(),
        "description": trafficinfo.get("description"),
        "status": trafficinfo.get("status"),  # 1: 畅通 2: 缓行 3: 拥堵 4: 严重拥堵
        "evaluation": trafficinfo.get("evaluation", {}),
    }

    return result


def get_yangzhou_weather():
    """获取扬州实时天气"""
    print("📡 获取扬州实时天气数据...")

    data = fetch_weather(YANGZHOU_CONFIG["amap_city_code"])
    parsed = parse_weather_data(data)

    if parsed:
        print(f"   ✅ 城市: {parsed.get('city')}")
        print(f"   报告时间: {parsed.get('report_time')}")
        forecasts = parsed.get('forecasts', [])
        if forecasts:
            today = forecasts[0]
            print(f"   今日天气: {today.get('dayweather')} / {today.get('nightweather')}")
            print(f"   温度: {today.get('nighttemp')}°C ~ {today.get('daytemp')}°C")

    return parsed


def get_yangzhou_traffic():
    """获取扬州交通状况"""
    print("\n📡 获取扬州交通态势数据...")

    location = f"{YANGZHOU_CONFIG['longitude']},{YANGZHOU_CONFIG['latitude']}"
    data = fetch_traffic(location)
    parsed = parse_traffic_data(data)

    if parsed:
        status_map = {"1": "畅通", "2": "缓行", "3": "拥堵", "4": "严重拥堵"}
        status = status_map.get(parsed.get('status'), "未知")
        print(f"   ✅ 交通状态: {status}")
        print(f"   描述: {parsed.get('description')}")

    return parsed


def save_realtime_data(weather_data, traffic_data):
    """保存实时数据到文件（追加模式）"""
    import json

    # 保存天气数据
    if weather_data:
        file_path = RAW_DATA_DIR / "realtime" / "amap_weather.jsonl"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(weather_data, ensure_ascii=False) + '\n')

    # 保存交通数据
    if traffic_data:
        file_path = RAW_DATA_DIR / "realtime" / "amap_traffic.jsonl"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(traffic_data, ensure_ascii=False) + '\n')


def main():
    """主函数"""
    print("=" * 60)
    print("  高德地图实时数据采集")
    print("=" * 60)

    # 获取天气数据
    weather = get_yangzhou_weather()

    # 获取交通数据
    traffic = get_yangzhou_traffic()

    # 保存数据
    save_realtime_data(weather, traffic)

    print("\n✅ 数据采集完成")

    return weather, traffic


if __name__ == "__main__":
    main()
