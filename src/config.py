# -*- coding: utf-8 -*-
"""
扬州空气质量预测系统 - 配置文件
"""

import os
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 中国时区 (UTC+8)
CHINA_TIMEZONE = timezone(timedelta(hours=8))

def get_china_now():
    """获取当前中国时间"""
    return datetime.now(CHINA_TIMEZONE)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"

# 模型目录
MODELS_DIR = PROJECT_ROOT / "models"

# 文档目录
DOCS_DIR = PROJECT_ROOT / "docs"

# Dashboard 目录
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DASHBOARD_DATA_DIR = DASHBOARD_DIR / "data"

# API Keys
WAQI_TOKEN = "37062d6027ab6b337cfed2ec15874c463571598a"
AMAP_KEY = "ce2297c0d9d5de44bf8faa26656da479"

# 扬州地理参数
YANGZHOU_CONFIG = {
    "name": "扬州",
    "latitude": 32.3912,
    "longitude": 119.4363,
    "amap_city_code": "321000",
    "qweather_location_id": "101190601",

    # 上风向城市（用于跨区域污染传输特征）
    "upwind_cities": {
        "南京": {"lat": 32.0603, "lon": 118.7969},
        "镇江": {"lat": 32.1879, "lon": 119.4253},
        "泰州": {"lat": 32.4906, "lon": 119.9232},
        "南通": {"lat": 31.9829, "lon": 120.8873}
    }
}

# 数据采集配置
DATA_COLLECTION_CONFIG = {
    # 历史数据时间范围
    # 完整数据: "2023-01-01" ~ "2026-02-02"
    # 测试用: 使用最近一个月
    "historical_start_date": "2025-01-01",
    "historical_end_date": "2026-02-02",  # quotsoft.net 有 1-2 天延迟

    # Open-Meteo 气象参数
    "weather_params": [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
        "precipitation",
        "cloud_cover"
    ]
}

# AQI 等级定义
AQI_LEVELS = {
    (0, 50): ("优", "green"),
    (51, 100): ("良", "yellow"),
    (101, 150): ("轻度污染", "orange"),
    (151, 200): ("中度污染", "red"),
    (201, 300): ("重度污染", "purple"),
    (301, 500): ("严重污染", "maroon")
}

def get_aqi_level(aqi):
    """根据 AQI 值返回等级"""
    for (low, high), (level, color) in AQI_LEVELS.items():
        if low <= aqi <= high:
            return level, color
    return "未知", "gray"

# 确保目录存在
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR,
                 MODELS_DIR, DOCS_DIR, DASHBOARD_DATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 历史数据子目录
(RAW_DATA_DIR / "historical_aqi").mkdir(exist_ok=True)
(RAW_DATA_DIR / "historical_weather").mkdir(exist_ok=True)
