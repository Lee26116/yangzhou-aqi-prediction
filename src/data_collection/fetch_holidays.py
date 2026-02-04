# -*- coding: utf-8 -*-
"""
获取中国节假日数据
从 GitHub 获取法定节假日和调休信息
"""

import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import RAW_DATA_DIR, DATA_COLLECTION_CONFIG

# GitHub 节假日数据源
HOLIDAY_API_URL = "https://raw.githubusercontent.com/lanceliao/china-holiday-calender/master/holidayAPI.json"


def fetch_holiday_data():
    """
    从 GitHub 获取节假日数据

    Returns:
        dict: 节假日数据
    """
    print("📥 下载中国节假日数据...")

    try:
        response = requests.get(HOLIDAY_API_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        print(f"   ✅ 获取成功")
        return data
    except Exception as e:
        print(f"   ❌ 获取失败: {e}")
        return None


def parse_holiday_data(data):
    """
    解析节假日数据，生成日期列表

    Args:
        data: API 返回的原始数据

    Returns:
        DataFrame: 包含日期和假日信息的表
    """
    if not data:
        return None

    records = []

    # 遍历数据
    # 格式通常是 {"2024": {"01-01": {"name": "元旦", "type": 1}, ...}, ...}
    # type: 1=假日, 2=调休工作日

    for year, dates in data.items():
        if not isinstance(dates, dict):
            continue

        for date_str, info in dates.items():
            if not isinstance(info, dict):
                continue

            try:
                full_date = f"{year}-{date_str}"
                date_obj = datetime.strptime(full_date, "%Y-%m-%d")

                records.append({
                    "date": date_obj.date(),
                    "holiday_name": info.get("name", ""),
                    "holiday_type": info.get("type", 0),  # 1=假日, 2=调休工作日
                    "is_holiday": info.get("type") == 1,
                    "is_workday_adjusted": info.get("type") == 2,  # 调休工作日
                })
            except:
                continue

    df = pd.DataFrame(records)

    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def generate_calendar(start_date, end_date):
    """
    生成完整的日历数据，包含周末和节假日信息

    Args:
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD

    Returns:
        DataFrame: 日历数据
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # 生成日期序列
    dates = []
    current = start
    while current <= end:
        dates.append({
            "date": current.date(),
            "year": current.year,
            "month": current.month,
            "day": current.day,
            "day_of_week": current.weekday(),  # 0=周一, 6=周日
            "day_name": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][current.weekday()],
            "is_weekend": current.weekday() >= 5,
        })
        current += timedelta(days=1)

    df = pd.DataFrame(dates)
    return df


def create_workday_calendar():
    """
    创建包含工作日信息的完整日历

    Returns:
        DataFrame
    """
    # 获取节假日数据
    holiday_data = fetch_holiday_data()
    holiday_df = parse_holiday_data(holiday_data)

    # 生成基础日历
    start_date = DATA_COLLECTION_CONFIG["historical_start_date"]
    end_date = DATA_COLLECTION_CONFIG["historical_end_date"]

    calendar_df = generate_calendar(start_date, end_date)

    # 合并节假日信息
    if holiday_df is not None and not holiday_df.empty:
        calendar_df = calendar_df.merge(
            holiday_df[["date", "holiday_name", "is_holiday", "is_workday_adjusted"]],
            on="date",
            how="left"
        )

        # 填充空值
        calendar_df["holiday_name"] = calendar_df["holiday_name"].fillna("")
        calendar_df["is_holiday"] = calendar_df["is_holiday"].fillna(False)
        calendar_df["is_workday_adjusted"] = calendar_df["is_workday_adjusted"].fillna(False)
    else:
        calendar_df["holiday_name"] = ""
        calendar_df["is_holiday"] = False
        calendar_df["is_workday_adjusted"] = False

    # 计算是否工作日
    # 工作日 = (不是周末 或 是调休工作日) 且 不是法定假日
    calendar_df["is_workday"] = (
        (~calendar_df["is_weekend"] | calendar_df["is_workday_adjusted"]) &
        ~calendar_df["is_holiday"]
    )

    # 添加季节
    def get_season(month):
        if month in [3, 4, 5]:
            return 1  # 春
        elif month in [6, 7, 8]:
            return 2  # 夏
        elif month in [9, 10, 11]:
            return 3  # 秋
        else:
            return 4  # 冬

    calendar_df["season"] = calendar_df["month"].apply(get_season)

    # 添加特殊时段标记
    # 秋收季节（秸秆焚烧）
    calendar_df["is_harvest_season"] = calendar_df["month"].isin([9, 10, 11])

    # 春节前后（烟花）
    def is_spring_festival_period(row):
        # 简单判断：农历新年通常在 1-2 月
        if row["month"] in [1, 2]:
            # 如果有春节假日标记
            if "春节" in str(row.get("holiday_name", "")):
                return True
            # 或者判断是否在春节假期附近
        return False

    calendar_df["is_spring_festival"] = calendar_df.apply(is_spring_festival_period, axis=1)

    return calendar_df


def main():
    """主函数"""
    print("=" * 60)
    print("  中国节假日数据处理")
    print("=" * 60)

    # 创建工作日历
    calendar_df = create_workday_calendar()

    # 保存
    output_file = RAW_DATA_DIR / "china_calendar.csv"
    calendar_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n✅ 日历数据已保存: {output_file}")
    print(f"   记录数: {len(calendar_df)}")
    print(f"   时间范围: {calendar_df['date'].min()} ~ {calendar_df['date'].max()}")
    print(f"   工作日: {calendar_df['is_workday'].sum()} 天")
    print(f"   法定假日: {calendar_df['is_holiday'].sum()} 天")
    print(f"   调休工作日: {calendar_df['is_workday_adjusted'].sum()} 天")

    return calendar_df


if __name__ == "__main__":
    main()
