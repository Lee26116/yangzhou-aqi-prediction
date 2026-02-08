# -*- coding: utf-8 -*-
"""
增量数据更新模块
每小时从 WAQI API 和 OpenMeteo API 获取最新数据
追加到 yangzhou_merged.csv 并重建 DL 特征
"""

import sys
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    PROCESSED_DATA_DIR, WAQI_TOKEN, YANGZHOU_CONFIG, CHINA_TIMEZONE
)

# WAQI 城市名映射（中文 → 拼音）
CITY_PINYIN = {
    '扬州': 'yangzhou',
    '南京': 'nanjing',
    '镇江': 'zhenjiang',
    '泰州': 'taizhou',
    '南通': 'nantong',
}

# OpenMeteo Forecast API（用于获取最近几小时的天气）
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_PARAMS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "precipitation",
    "cloud_cover", "weather_code", "dew_point_2m", "apparent_temperature",
    "rain", "snowfall", "visibility", "wind_gusts_10m",
    "soil_temperature_0cm", "boundary_layer_height", "uv_index"
]

# 中国 HJ 633-2012 IAQI 断点表：(IAQI_low, IAQI_high, C_low, C_high)
# WAQI API 返回 IAQI 子指数，需要转换回原始浓度以匹配 quotsoft.net 训练数据
IAQI_BREAKPOINTS = {
    # CO: mg/m³ (24h → 近似1h)
    'co': [
        (0, 50, 0, 5), (50, 100, 5, 10), (100, 150, 10, 35),
        (150, 200, 35, 60), (200, 300, 60, 90), (300, 400, 90, 120),
        (400, 500, 120, 150),
    ],
    # NO2: µg/m³ (1h)
    'no2': [
        (0, 50, 0, 100), (50, 100, 100, 200), (100, 150, 200, 700),
        (150, 200, 700, 1200), (200, 300, 1200, 2340), (300, 400, 2340, 3090),
        (400, 500, 3090, 3840),
    ],
    # SO2: µg/m³ (1h)
    'so2': [
        (0, 50, 0, 150), (50, 100, 150, 500), (100, 150, 500, 650),
        (150, 200, 650, 800), (200, 300, 800, 1600), (300, 400, 1600, 2100),
        (400, 500, 2100, 2620),
    ],
    # O3: µg/m³ (1h)
    'o3': [
        (0, 50, 0, 160), (50, 100, 160, 200), (100, 150, 200, 300),
        (150, 200, 300, 400), (200, 300, 400, 800), (300, 400, 800, 1000),
        (400, 500, 1000, 1200),
    ],
    # PM2.5: µg/m³ (24h → 近似1h)
    'pm25': [
        (0, 50, 0, 35), (50, 100, 35, 75), (100, 150, 75, 115),
        (150, 200, 115, 150), (200, 300, 150, 250), (300, 400, 250, 350),
        (400, 500, 350, 500),
    ],
    # PM10: µg/m³ (24h → 近似1h)
    'pm10': [
        (0, 50, 0, 50), (50, 100, 50, 150), (100, 150, 150, 250),
        (150, 200, 250, 350), (200, 300, 350, 420), (300, 400, 420, 500),
        (400, 500, 500, 600),
    ],
}


def iaqi_to_concentration(iaqi_value, pollutant):
    """
    将 WAQI IAQI 子指数转换为原始浓度值

    WAQI API 对中国站点返回的是 HJ 633-2012 标准的 IAQI 子指数，
    而训练数据 (quotsoft.net) 使用原始浓度 (CO: mg/m³, 其他: µg/m³)。
    """
    if iaqi_value is None or pollutant not in IAQI_BREAKPOINTS:
        return iaqi_value

    breakpoints = IAQI_BREAKPOINTS[pollutant]
    for iaqi_lo, iaqi_hi, c_lo, c_hi in breakpoints:
        if iaqi_lo <= iaqi_value <= iaqi_hi:
            ratio = (iaqi_value - iaqi_lo) / (iaqi_hi - iaqi_lo)
            return round(c_lo + ratio * (c_hi - c_lo), 2)

    # 超出范围，用最高断点外推
    if iaqi_value > 500:
        last = breakpoints[-1]
        return last[3]  # 返回最高浓度
    return iaqi_value


def fetch_waqi(city_pinyin):
    """从 WAQI API 获取城市实时 AQI 数据，并转换为原始浓度"""
    url = f"https://api.waqi.info/feed/{city_pinyin}/?token={WAQI_TOKEN}"
    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()
        if data.get('status') != 'ok':
            return None

        d = data['data']
        iaqi = d.get('iaqi', {})

        def v(key):
            val = iaqi.get(key)
            return val.get('v') if isinstance(val, dict) else val

        # 获取原始 IAQI 子指数
        raw_pm25 = v('pm25')
        raw_pm10 = v('pm10')
        raw_no2 = v('no2')
        raw_o3 = v('o3')
        raw_co = v('co')
        raw_so2 = v('so2')

        # 转换为原始浓度
        return {
            'aqi': d.get('aqi'),
            'pm25': iaqi_to_concentration(raw_pm25, 'pm25'),
            'pm10': iaqi_to_concentration(raw_pm10, 'pm10'),
            'no2': iaqi_to_concentration(raw_no2, 'no2'),
            'o3': iaqi_to_concentration(raw_o3, 'o3'),
            'co': iaqi_to_concentration(raw_co, 'co'),
            'so2': iaqi_to_concentration(raw_so2, 'so2'),
            'time_iso': d.get('time', {}).get('iso'),
        }
    except Exception as e:
        print(f"   ⚠️ WAQI {city_pinyin} 失败: {e}")
        return None


def fetch_openmeteo_recent(lat, lon, past_hours=6):
    """从 OpenMeteo Forecast API 获取最近几小时的天气数据"""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_PARAMS),
        "past_hours": past_hours,
        "forecast_hours": 0,
        "timezone": "Asia/Shanghai",
    }
    try:
        resp = requests.get(FORECAST_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "hourly" not in data:
            return None

        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        df['datetime'] = pd.to_datetime(df['time'])
        df = df.drop(columns=['time'])
        return df

    except Exception as e:
        print(f"   ⚠️ OpenMeteo 失败: {e}")
        return None


def compute_time_features(dt):
    """计算时间特征"""
    hour = dt.hour
    day = dt.day
    month = dt.month
    year = dt.year
    dow = dt.weekday()

    return {
        'hour': hour,
        'day': day,
        'month': month,
        'year': year,
        'day_of_week': dow,
        'day_of_year': dt.timetuple().tm_yday,
        'week_of_year': dt.isocalendar()[1],
        'is_weekend': dow >= 5,
        'is_holiday': False,
        'is_workday': dow < 5,
        'season': 1 if month in [3, 4, 5] else (2 if month in [6, 7, 8] else (3 if month in [9, 10, 11] else 4)),
        'is_harvest_season': month in [6, 7, 10, 11],
        'holiday_name': np.nan,
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'day_of_week_sin': np.sin(2 * np.pi * dow / 7),
        'day_of_week_cos': np.cos(2 * np.pi * dow / 7),
    }


def update_merged_data():
    """增量更新 yangzhou_merged.csv"""
    merged_file = PROCESSED_DATA_DIR / "yangzhou_merged.csv"
    if not merged_file.exists():
        print("❌ yangzhou_merged.csv 不存在")
        return False

    # 读取现有数据
    df = pd.read_csv(merged_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    last_dt = df['datetime'].max()
    china_now = datetime.now(CHINA_TIMEZONE).replace(
        minute=0, second=0, microsecond=0, tzinfo=None
    )

    # 计算需要补充的小时数
    hours_gap = int((china_now - last_dt).total_seconds() / 3600)
    if hours_gap <= 0:
        print(f"   数据已是最新 (最后: {last_dt})")
        return True

    print(f"   最后数据: {last_dt}")
    print(f"   当前时间: {china_now}")
    print(f"   需补充: {hours_gap} 小时")

    # 获取扬州 AQI
    print("\n📡 获取扬州实时 AQI...")
    yz_aqi = fetch_waqi('yangzhou')
    if yz_aqi:
        print(f"   AQI={yz_aqi['aqi']}, PM2.5={yz_aqi['pm25']}, PM10={yz_aqi['pm10']}")
    else:
        print("   ⚠️ 获取失败，使用 NaN")

    # 获取上风向城市 AQI
    print("\n📡 获取上风向城市 AQI...")
    upwind_aqi = {}
    for city_cn, city_py in CITY_PINYIN.items():
        if city_cn == '扬州':
            continue
        data = fetch_waqi(city_py)
        if data:
            upwind_aqi[city_cn] = data
            print(f"   {city_cn}: AQI={data['aqi']}")
        else:
            print(f"   {city_cn}: ⚠️ 失败")

    # 获取天气数据
    print(f"\n🌤️ 获取最近 {min(hours_gap + 2, 48)} 小时天气...")
    weather_df = fetch_openmeteo_recent(
        YANGZHOU_CONFIG['latitude'],
        YANGZHOU_CONFIG['longitude'],
        past_hours=min(hours_gap + 2, 48)
    )
    if weather_df is not None:
        print(f"   获取到 {len(weather_df)} 条天气记录")
    else:
        print("   ⚠️ 天气数据获取失败")

    # 获取最后已知的 AQI 值（用于线性插值填充缺口）
    last_known = {}
    for col in ['AQI', 'PM2.5', 'PM10', 'NO2', 'O3', 'CO', 'SO2']:
        vals = df[col].dropna()
        last_known[col] = vals.iloc[-1] if len(vals) > 0 else None

    # 上风向城市最后已知值
    for city_cn in ['南京', '镇江', '泰州', '南通']:
        for suffix in ['AQI', 'PM2.5', 'PM10']:
            col = f'{city_cn}_{suffix}'
            if col in df.columns:
                vals = df[col].dropna()
                last_known[col] = vals.iloc[-1] if len(vals) > 0 else None

    # 构建新行
    new_rows = []
    for h in range(1, hours_gap + 1):
        dt = last_dt + timedelta(hours=h)
        if dt > china_now:
            break

        row = {'datetime': dt}

        # AQI 数据：线性插值从最后已知值过渡到当前实时值
        # 这样模型看到的是平滑变化而非突变
        ratio = h / hours_gap  # 0→1，从旧值过渡到当前值

        if yz_aqi:
            for col, waqi_key in [('AQI', 'aqi'), ('PM2.5', 'pm25'),
                                   ('PM10', 'pm10'), ('NO2', 'no2'),
                                   ('O3', 'o3'), ('CO', 'co'), ('SO2', 'so2')]:
                old_val = last_known.get(col)
                new_val = yz_aqi.get(waqi_key)
                if old_val is not None and new_val is not None:
                    row[col] = old_val * (1 - ratio) + new_val * ratio
                elif new_val is not None:
                    row[col] = new_val

            # 上风向城市：同样线性插值
            for city_cn, data in upwind_aqi.items():
                for suffix, waqi_key in [('AQI', 'aqi'), ('PM2.5', 'pm25'), ('PM10', 'pm10')]:
                    col = f'{city_cn}_{suffix}'
                    old_val = last_known.get(col)
                    new_val = data.get(waqi_key)
                    if old_val is not None and new_val is not None:
                        row[col] = old_val * (1 - ratio) + new_val * ratio
                    elif new_val is not None:
                        row[col] = new_val

        # 天气数据
        if weather_df is not None:
            weather_row = weather_df[weather_df['datetime'] == dt]
            if len(weather_row) > 0:
                wr = weather_row.iloc[0]
                for col in WEATHER_PARAMS:
                    if col in wr.index:
                        row[col] = wr[col]

        # 时间特征
        row.update(compute_time_features(dt))

        new_rows.append(row)

    if not new_rows:
        print("   无新数据需要添加")
        return True

    # 追加到 DataFrame
    new_df = pd.DataFrame(new_rows)
    # 确保列对齐
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = np.nan

    new_df = new_df[df.columns]  # 保持列顺序
    df = pd.concat([df, new_df], ignore_index=True)

    # 去重
    df = df.drop_duplicates(subset=['datetime'], keep='last').sort_values('datetime').reset_index(drop=True)

    # 计算 _24h 滚动均值（对最近的行）
    for col in ['PM2.5', 'PM10', 'CO', 'NO2', 'O3', 'SO2']:
        col_24h = f'{col}_24h'
        if col_24h in df.columns and col in df.columns:
            df[col_24h] = df[col].rolling(window=24, min_periods=1).mean()

    for city_cn in ['南京', '镇江', '泰州', '南通']:
        for pollutant in ['PM2.5', 'PM10']:
            col = f'{city_cn}_{pollutant}'
            col_24h = f'{city_cn}_{pollutant}_24h'
            if col_24h in df.columns and col in df.columns:
                df[col_24h] = df[col].rolling(window=24, min_periods=1).mean()

    # O3_8h 和 O3_8h_24h
    if 'O3' in df.columns:
        df['O3_8h'] = df['O3'].rolling(window=8, min_periods=1).mean()
        if 'O3_8h_24h' in df.columns:
            df['O3_8h_24h'] = df['O3_8h'].rolling(window=24, min_periods=1).mean()

    # 保存
    df.to_csv(merged_file, index=False, encoding='utf-8')
    print(f"\n✅ 已追加 {len(new_rows)} 行到 {merged_file}")
    print(f"   数据范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    print(f"   总行数: {len(df)}")

    return True


def rebuild_dl_features():
    """重建 DL 特征文件"""
    print("\n🔧 重建 DL 特征...")
    try:
        from deep_learning.data.build_dl_features import main as build_features
        build_features()
        return True
    except Exception as e:
        print(f"   ⚠️ DL 特征构建失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("  增量数据更新")
    print("=" * 60)

    success = update_merged_data()
    if success:
        rebuild_dl_features()

    print("\n✅ 增量更新完成")


if __name__ == "__main__":
    main()
