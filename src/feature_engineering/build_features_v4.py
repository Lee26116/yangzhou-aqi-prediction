# -*- coding: utf-8 -*-
"""
特征工程 v4 - 兼顾方法论正确性和可解释性

设计原则：
1. 方法论正确：预测 h 小时后，只使用 h 小时前及更早的数据
2. 可解释性优先：包含所有环境科学上公认与空气质量相关的变量
3. 领域知识驱动：即使某变量统计重要性不高，只要物理意义明确，就保留

环境科学上与 AQI 相关的因素：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
因素类别        | 变量                | 与 AQI 的关系        | 物理机制
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
气象-扩散条件    | 风速                | 负相关 (-)          | 高风速促进扩散
               | 边界层高度           | 负相关 (-)          | 高边界层扩散空间大
               | 气压                | 正相关 (+)          | 高压伴随稳定天气
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
气象-化学反应    | 温度                | 正相关 (+)          | 高温加速光化学反应
               | 湿度                | 正相关 (+)          | 高湿度促进二次颗粒物
               | 露点温度             | 正相关 (+)          | 反映大气湿度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
气象-清除机制    | 降水                | 负相关 (-)          | 雨水冲刷污染物
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
时间-排放模式    | 小时                | 早晚高峰高           | 交通排放
               | 工作日/周末          | 工作日略高           | 工业和交通
               | 供暖季              | 冬季高              | 供暖排放
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
污染物          | PM2.5              | 直接组成            | AQI 主要成分
               | PM10               | 直接组成            | 粗颗粒物
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PROCESSED_DATA_DIR, FEATURES_DATA_DIR


# 定义可解释特征及其物理意义
INTERPRETABLE_FEATURES = {
    # ========== 气象因素 - 扩散条件 ==========
    'wind_speed_10m': {
        'category': '气象-扩散',
        'expected_sign': '-',  # 负相关：高风速有利于扩散
        'physical_meaning': '风速越大，污染物扩散越快，AQI 越低',
        'unit': 'm/s'
    },
    'surface_pressure': {
        'category': '气象-扩散',
        'expected_sign': '+',  # 正相关：高压伴随稳定天气
        'physical_meaning': '高气压通常伴随下沉气流和稳定层结，不利于扩散',
        'unit': 'hPa'
    },

    # ========== 气象因素 - 化学反应 ==========
    'temperature_2m': {
        'category': '气象-化学',
        'expected_sign': '+/-',  # 复杂关系
        'physical_meaning': '高温加速光化学反应生成O3，但也促进对流扩散',
        'unit': '°C'
    },
    'relative_humidity_2m': {
        'category': '气象-化学',
        'expected_sign': '+',  # 正相关
        'physical_meaning': '高湿度促进二次颗粒物生成，增加 PM2.5',
        'unit': '%'
    },
    'dew_point_2m': {
        'category': '气象-化学',
        'expected_sign': '+',  # 正相关
        'physical_meaning': '露点反映大气湿度，高露点利于颗粒物吸湿增长',
        'unit': '°C'
    },

    # ========== 污染物 ==========
    'PM2.5': {
        'category': '污染物',
        'expected_sign': '+',
        'physical_meaning': 'PM2.5 是 AQI 的主要组成部分',
        'unit': 'μg/m³'
    },
    'PM10': {
        'category': '污染物',
        'expected_sign': '+',
        'physical_meaning': 'PM10 包含粗颗粒物，也是 AQI 成分',
        'unit': 'μg/m³'
    },
}

# 定义时间相关特征
TIME_FEATURES = {
    'hour': {
        'category': '时间-排放',
        'physical_meaning': '早晚高峰（7-9, 17-19）排放增加',
    },
    'is_weekend': {
        'category': '时间-排放',
        'physical_meaning': '周末工业和交通减少，排放略低',
    },
    'is_rush_hour': {
        'category': '时间-排放',
        'physical_meaning': '早晚高峰期交通排放增加',
    },
    'is_heating_season': {
        'category': '时间-排放',
        'physical_meaning': '供暖季（11-3月）燃煤排放增加',
    },
}


def build_interpretable_features(df, horizon_hours):
    """
    构建可解释特征

    所有特征都有明确的物理意义，便于解释模型结果
    """
    print(f"\n🔧 构建可解释特征 ({horizon_hours}h 预测)...")
    print(f"   原则: 每个特征都必须有明确的物理意义")

    result = df[['datetime']].copy()
    result['AQI_target'] = df['AQI']  # 目标变量

    feature_descriptions = []  # 记录特征描述

    # ========== 1. 历史 AQI 特征 ==========
    print(f"\n   [1] 历史 AQI (时间惯性)")

    # 预测时刻可知的最近 AQI（使用明确的 lag 命名避免混淆）
    result[f'AQI_lag_{horizon_hours}h'] = df['AQI'].shift(horizon_hours)
    feature_descriptions.append({
        'feature': f'AQI_lag_{horizon_hours}h',
        'category': 'AQI历史',
        'meaning': f'{horizon_hours}小时前的AQI值（预测时刻可知的最新历史值，不是目标变量！）',
        'expected_sign': '+',
        'importance': '高'
    })

    # 昨天同一时刻
    result['AQI_yesterday_same_hour'] = df['AQI'].shift(24 + horizon_hours)
    feature_descriptions.append({
        'feature': 'AQI_yesterday_same_hour',
        'category': 'AQI历史',
        'meaning': '昨天同一时刻的AQI，反映日周期性',
        'expected_sign': '+',
        'importance': '中'
    })

    # 24小时滚动均值
    shifted = df['AQI'].shift(horizon_hours)
    result['AQI_rolling_mean_24h'] = shifted.rolling(24, min_periods=1).mean()
    feature_descriptions.append({
        'feature': 'AQI_rolling_mean_24h',
        'category': 'AQI历史',
        'meaning': '过去24小时平均AQI，反映整体污染水平',
        'expected_sign': '+',
        'importance': '高'
    })

    # 24小时变化趋势
    result['AQI_trend_24h'] = result[f'AQI_lag_{horizon_hours}h'] - df['AQI'].shift(24 + horizon_hours)
    feature_descriptions.append({
        'feature': 'AQI_trend_24h',
        'category': 'AQI历史',
        'meaning': '过去24小时AQI变化量，正值表示恶化趋势',
        'expected_sign': '+',
        'importance': '中'
    })

    # ========== 2. 污染物特征 ==========
    print(f"   [2] 污染物浓度")

    for pollutant in ['PM2.5', 'PM10']:
        if pollutant in df.columns:
            col_name = f'{pollutant}_current'
            result[col_name] = df[pollutant].shift(horizon_hours)
            info = INTERPRETABLE_FEATURES.get(pollutant, {})
            feature_descriptions.append({
                'feature': col_name,
                'category': '污染物',
                'meaning': info.get('physical_meaning', f'{pollutant}浓度'),
                'expected_sign': '+',
                'importance': '高'
            })

    # ========== 3. 气象因素 - 扩散条件 ==========
    print(f"   [3] 气象-扩散条件 (风速、气压)")

    # 风速
    if 'wind_speed_10m' in df.columns:
        result['wind_speed'] = df['wind_speed_10m'].shift(horizon_hours)
        feature_descriptions.append({
            'feature': 'wind_speed',
            'category': '气象-扩散',
            'meaning': '风速：高风速促进污染物扩散，降低AQI',
            'expected_sign': '-',
            'importance': '高'
        })

        # 低风速标记（不利于扩散）
        result['is_low_wind'] = (result['wind_speed'] < 2).astype(int)
        feature_descriptions.append({
            'feature': 'is_low_wind',
            'category': '气象-扩散',
            'meaning': '低风速标记：风速<2m/s时不利于扩散',
            'expected_sign': '+',
            'importance': '中'
        })

    # 气压
    if 'surface_pressure' in df.columns:
        result['pressure'] = df['surface_pressure'].shift(horizon_hours)
        feature_descriptions.append({
            'feature': 'pressure',
            'category': '气象-扩散',
            'meaning': '气压：高压伴随稳定天气，不利于扩散',
            'expected_sign': '+',
            'importance': '中'
        })

        # 气压变化（天气系统变化）
        result['pressure_change_24h'] = result['pressure'] - df['surface_pressure'].shift(24 + horizon_hours)
        feature_descriptions.append({
            'feature': 'pressure_change_24h',
            'category': '气象-扩散',
            'meaning': '气压变化：气压下降通常伴随天气系统过境',
            'expected_sign': '-',
            'importance': '中'
        })

    # ========== 4. 气象因素 - 化学反应 ==========
    print(f"   [4] 气象-化学反应 (温度、湿度)")

    # 温度
    if 'temperature_2m' in df.columns:
        result['temperature'] = df['temperature_2m'].shift(horizon_hours)
        feature_descriptions.append({
            'feature': 'temperature',
            'category': '气象-化学',
            'meaning': '温度：高温加速光化学反应，但也促进对流',
            'expected_sign': '+/-',
            'importance': '中'
        })

        # 日温差（逆温层代理）
        # 日温差小说明大气层结稳定，可能有逆温
        daily_range = df.groupby(df['datetime'].dt.date)['temperature_2m'].transform(
            lambda x: x.max() - x.min()
        )
        result['temp_daily_range'] = daily_range.shift(horizon_hours)
        feature_descriptions.append({
            'feature': 'temp_daily_range',
            'category': '气象-扩散',
            'meaning': '日温差：温差小表示大气稳定，可能有逆温层',
            'expected_sign': '-',
            'importance': '中'
        })

    # 湿度
    if 'relative_humidity_2m' in df.columns:
        result['humidity'] = df['relative_humidity_2m'].shift(horizon_hours)
        feature_descriptions.append({
            'feature': 'humidity',
            'category': '气象-化学',
            'meaning': '相对湿度：高湿度促进二次颗粒物生成',
            'expected_sign': '+',
            'importance': '高'
        })

        # 高湿度标记
        result['is_high_humidity'] = (result['humidity'] > 80).astype(int)
        feature_descriptions.append({
            'feature': 'is_high_humidity',
            'category': '气象-化学',
            'meaning': '高湿度标记：湿度>80%时二次颗粒物生成加速',
            'expected_sign': '+',
            'importance': '中'
        })

    # 露点温度
    if 'dew_point_2m' in df.columns:
        result['dew_point'] = df['dew_point_2m'].shift(horizon_hours)
        feature_descriptions.append({
            'feature': 'dew_point',
            'category': '气象-化学',
            'meaning': '露点温度：反映大气绝对湿度',
            'expected_sign': '+',
            'importance': '中'
        })

    # ========== 5. 时间特征 - 排放模式 ==========
    print(f"   [5] 时间-排放模式")

    # 小时
    result['hour'] = df['datetime'].dt.hour
    feature_descriptions.append({
        'feature': 'hour',
        'category': '时间-排放',
        'meaning': '一天中的小时：反映人类活动周期',
        'expected_sign': '非线性',
        'importance': '中'
    })

    # 周期编码
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)

    # 早晚高峰
    result['is_rush_hour'] = (
        ((result['hour'] >= 7) & (result['hour'] <= 9)) |
        ((result['hour'] >= 17) & (result['hour'] <= 19))
    ).astype(int)
    feature_descriptions.append({
        'feature': 'is_rush_hour',
        'category': '时间-排放',
        'meaning': '早晚高峰：7-9点和17-19点交通排放增加',
        'expected_sign': '+',
        'importance': '中'
    })

    # 周末
    result['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
    feature_descriptions.append({
        'feature': 'is_weekend',
        'category': '时间-排放',
        'meaning': '周末：工业和交通活动减少',
        'expected_sign': '-',
        'importance': '低'
    })

    # 供暖季
    month = df['datetime'].dt.month
    result['is_heating_season'] = month.isin([11, 12, 1, 2, 3]).astype(int)
    feature_descriptions.append({
        'feature': 'is_heating_season',
        'category': '时间-排放',
        'meaning': '供暖季（11-3月）：北方燃煤供暖增加排放',
        'expected_sign': '+',
        'importance': '高'
    })

    # 月份周期编码
    result['month_sin'] = np.sin(2 * np.pi * month / 12)
    result['month_cos'] = np.cos(2 * np.pi * month / 12)
    feature_descriptions.append({
        'feature': 'month_sin/cos',
        'category': '时间-季节',
        'meaning': '季节周期：捕捉季节性变化',
        'expected_sign': '周期性',
        'importance': '中'
    })

    # 清理临时列
    result = result.drop(columns=['hour'], errors='ignore')

    # 保存特征描述
    desc_df = pd.DataFrame(feature_descriptions)

    print(f"\n   ✅ 共构建 {len([c for c in result.columns if c not in ['datetime', 'AQI_target']])} 个可解释特征")

    return result, desc_df


def validate_feature_signs(df, desc_df):
    """
    验证特征与目标变量的相关性是否符合预期

    这是可解释性的关键：确保模型学到的关系符合物理规律
    """
    print(f"\n🔬 验证特征相关性与预期符号...")

    target = 'AQI_target'
    results = []

    for _, row in desc_df.iterrows():
        feat = row['feature']
        if feat not in df.columns or '/' in feat:  # 跳过组合特征
            continue

        expected = row['expected_sign']
        corr = df[feat].corr(df[target])

        # 判断实际符号
        if corr > 0.05:
            actual = '+'
        elif corr < -0.05:
            actual = '-'
        else:
            actual = '~0'

        # 判断是否符合预期
        if expected in ['+/-', '非线性', '周期性']:
            match = '✓'  # 复杂关系，不做判断
        elif expected == actual:
            match = '✓'
        elif actual == '~0':
            match = '○'  # 相关性很弱
        else:
            match = '✗'  # 符号相反，需要关注

        results.append({
            'feature': feat,
            'category': row['category'],
            'expected': expected,
            'actual': actual,
            'correlation': corr,
            'match': match
        })

    results_df = pd.DataFrame(results)

    # 打印结果
    print(f"\n   {'特征':<25} {'类别':<12} {'预期':<6} {'实际':<6} {'相关系数':<10} {'符合'}")
    print("   " + "-" * 75)

    for _, r in results_df.iterrows():
        print(f"   {r['feature']:<25} {r['category']:<12} {r['expected']:<6} {r['actual']:<6} {r['correlation']:>8.3f}   {r['match']}")

    # 统计
    n_match = (results_df['match'] == '✓').sum()
    n_total = len(results_df)
    print(f"\n   符合预期: {n_match}/{n_total} ({100*n_match/n_total:.0f}%)")

    return results_df


def main():
    """主函数"""
    print("=" * 70)
    print("  特征工程 v4 - 可解释性优先")
    print("  原则: 每个特征都有明确的物理意义，便于解释模型预测")
    print("=" * 70)

    # 读取数据
    input_file = PROCESSED_DATA_DIR / "yangzhou_merged.csv"
    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return

    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"\n📂 读取数据: {len(df)} 条")

    # 为不同预测时间跨度构建特征
    horizons = [1, 6, 12, 24]

    for h in horizons:
        print(f"\n{'='*70}")
        print(f"  预测时间跨度: {h} 小时")
        print(f"{'='*70}")

        # 构建可解释特征
        features_df, desc_df = build_interpretable_features(df, h)

        # 删除缺失目标的行
        features_df = features_df.dropna(subset=['AQI_target'])
        print(f"\n   有效样本数: {len(features_df)}")

        # 验证特征符号
        validation_df = validate_feature_signs(features_df, desc_df)

        # 保存特征数据
        output_file = FEATURES_DATA_DIR / f"yangzhou_features_v4_{h}h.csv"
        features_df.to_csv(output_file, index=False)
        print(f"\n💾 特征数据已保存: {output_file}")

        # 保存特征描述（用于 Dashboard 展示）
        desc_file = FEATURES_DATA_DIR / f"feature_descriptions_v4_{h}h.csv"
        desc_df.to_csv(desc_file, index=False)
        print(f"💾 特征描述已保存: {desc_file}")

    print("\n" + "=" * 70)
    print("  特征工程 v4 完成")
    print("  所有特征都有明确的物理意义，可用于解释模型预测")
    print("=" * 70)


if __name__ == "__main__":
    main()
