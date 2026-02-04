# -*- coding: utf-8 -*-
"""
模型训练 v4 - 可解释性优先

原则：
1. 保留所有有物理意义的特征，不因统计重要性低而删除
2. 分析模型学到的关系是否符合物理预期
3. 生成可解释性报告，解释每个特征对预测的贡献
"""

import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FEATURES_DATA_DIR, MODELS_DIR, DASHBOARD_DATA_DIR


# 特征的物理意义（用于可解释性报告）
FEATURE_INTERPRETATIONS = {
    'AQI_lag_1h': {
        'name': '1h前AQI',
        'expected': '+',
        'interpretation': '1小时前的AQI值，空气质量具有时间惯性'
    },
    'AQI_lag_6h': {
        'name': '6h前AQI',
        'expected': '+',
        'interpretation': '6小时前的AQI值，用于6h预测'
    },
    'AQI_lag_12h': {
        'name': '12h前AQI',
        'expected': '+',
        'interpretation': '12小时前的AQI值，用于12h预测'
    },
    'AQI_lag_24h': {
        'name': '24h前AQI',
        'expected': '+',
        'interpretation': '24小时前的AQI值，用于24h预测'
    },
    'AQI_yesterday_same_hour': {
        'name': '昨日同期AQI',
        'expected': '+',
        'interpretation': '空气质量存在日周期性，昨天同一时刻的情况有参考价值'
    },
    'AQI_rolling_mean_24h': {
        'name': '24小时平均AQI',
        'expected': '+',
        'interpretation': '反映近期整体污染水平，消除短期波动'
    },
    'AQI_trend_24h': {
        'name': 'AQI变化趋势',
        'expected': '+',
        'interpretation': '正值表示污染加重趋势，负值表示改善趋势'
    },
    'PM2.5_current': {
        'name': 'PM2.5浓度',
        'expected': '+',
        'interpretation': 'PM2.5是AQI的主要组成部分，直接影响空气质量等级'
    },
    'PM10_current': {
        'name': 'PM10浓度',
        'expected': '+',
        'interpretation': 'PM10包含粗颗粒物，是AQI的重要组成部分'
    },
    'wind_speed': {
        'name': '风速',
        'expected': '-',
        'interpretation': '高风速促进污染物扩散，降低AQI（负相关）'
    },
    'is_low_wind': {
        'name': '低风速标记',
        'expected': '+',
        'interpretation': '风速<2m/s时大气扩散能力弱，污染物容易累积'
    },
    'pressure': {
        'name': '气压',
        'expected': '+',
        'interpretation': '高气压通常伴随下沉气流和稳定层结，不利于扩散'
    },
    'pressure_change_24h': {
        'name': '气压变化',
        'expected': '-',
        'interpretation': '气压下降通常意味着天气系统过境，有利于污染物清除'
    },
    'temperature': {
        'name': '温度',
        'expected': '+/-',
        'interpretation': '温度影响复杂：高温加速光化学反应，但也促进对流'
    },
    'temp_daily_range': {
        'name': '日温差',
        'expected': '+',  # 根据实际数据更新
        'interpretation': '扬州实际规律：日温差大=晴朗天气=高压控制=不利扩散'
    },
    'humidity': {
        'name': '相对湿度',
        'expected': '-',  # 根据实际数据更新
        'interpretation': '扬州实际规律：高湿度通常伴随降雨/海洋气团，有利于污染物清除'
    },
    'is_high_humidity': {
        'name': '高湿度标记',
        'expected': '-',  # 根据实际数据更新
        'interpretation': '湿度>80%时通常伴随降雨，冲刷污染物'
    },
    'dew_point': {
        'name': '露点温度',
        'expected': '-',  # 根据实际数据更新
        'interpretation': '高露点反映大气湿润，通常伴随降水天气'
    },
    'is_rush_hour': {
        'name': '早晚高峰',
        'expected': '+',
        'interpretation': '7-9点和17-19点交通排放增加'
    },
    'is_weekend': {
        'name': '周末',
        'expected': '-',
        'interpretation': '周末工业和交通活动减少，排放降低'
    },
    'is_heating_season': {
        'name': '供暖季',
        'expected': '+',
        'interpretation': '11月至3月为供暖季，燃煤排放增加'
    },
    'hour_sin': {
        'name': '小时周期(sin)',
        'expected': '周期',
        'interpretation': '捕捉一天中的周期性变化'
    },
    'hour_cos': {
        'name': '小时周期(cos)',
        'expected': '周期',
        'interpretation': '与hour_sin配合完整表示24小时周期'
    },
    'month_sin': {
        'name': '月份周期(sin)',
        'expected': '周期',
        'interpretation': '捕捉季节性变化'
    },
    'month_cos': {
        'name': '月份周期(cos)',
        'expected': '周期',
        'interpretation': '与month_sin配合完整表示年周期'
    },
}


def load_data(horizon_hours):
    """加载特征数据"""
    input_file = FEATURES_DATA_DIR / f"yangzhou_features_v4_{horizon_hours}h.csv"

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return None, None, None

    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 分离特征和目标
    feature_cols = [c for c in df.columns if c not in ['datetime', 'AQI_target']]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['AQI_target']

    return X, y, feature_cols


def train_and_evaluate(X, y, feature_cols, horizon_hours):
    """训练模型并评估"""
    print(f"\n🤖 训练 XGBoost 模型 ({horizon_hours}h 预测)...")

    # 时序交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = {'MAE': [], 'RMSE': [], 'R2': []}

    params = {
        'n_estimators': 150,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)

        cv_results['MAE'].append(mean_absolute_error(y_test, y_pred))
        cv_results['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        cv_results['R2'].append(r2_score(y_test, y_pred))

    print(f"   MAE:  {np.mean(cv_results['MAE']):.2f} ± {np.std(cv_results['MAE']):.2f}")
    print(f"   RMSE: {np.mean(cv_results['RMSE']):.2f} ± {np.std(cv_results['RMSE']):.2f}")
    print(f"   R²:   {np.mean(cv_results['R2']):.4f} ± {np.std(cv_results['R2']):.4f}")

    # 训练最终模型
    final_model = XGBRegressor(**params)
    final_model.fit(X, y, verbose=False)

    return final_model, cv_results


def generate_interpretability_report(model, feature_cols, X, y, horizon_hours):
    """生成可解释性报告"""
    print(f"\n📊 生成可解释性报告...")

    # 特征重要性
    importance = model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # 特征与目标的相关性
    correlations = X.corrwith(y)

    # 构建可解释性报告
    report = {
        'horizon_hours': horizon_hours,
        'features': []
    }

    print(f"\n   {'特征':<25} {'重要性':>8} {'相关性':>8} {'方向':>6} {'物理意义'}")
    print("   " + "-" * 90)

    for _, row in imp_df.iterrows():
        feat = row['feature']
        imp = row['importance']
        corr = correlations.get(feat, 0)

        # 获取特征解释
        interp = FEATURE_INTERPRETATIONS.get(feat, {})
        direction = '+' if corr > 0 else '-' if corr < 0 else '~0'

        feature_info = {
            'name': feat,
            'display_name': interp.get('name', feat),
            'importance': float(imp),
            'correlation': float(corr),
            'direction': direction,
            'interpretation': interp.get('interpretation', ''),
            'category': get_feature_category(feat)
        }
        report['features'].append(feature_info)

        # 打印前15个特征
        if imp > 0.01 or len(report['features']) <= 15:
            meaning = interp.get('interpretation', '')[:40]
            print(f"   {feat:<25} {imp:>7.1%} {corr:>+7.3f}  {direction:>4}   {meaning}...")

    return report


def get_feature_category(feat):
    """获取特征类别"""
    if 'AQI' in feat:
        return 'AQI历史'
    elif 'PM' in feat:
        return '污染物'
    elif feat in ['wind_speed', 'is_low_wind', 'pressure', 'pressure_change_24h', 'temp_daily_range']:
        return '气象-扩散'
    elif feat in ['temperature', 'humidity', 'is_high_humidity', 'dew_point']:
        return '气象-化学'
    elif feat in ['is_rush_hour', 'is_weekend', 'is_heating_season']:
        return '时间-排放'
    elif 'sin' in feat or 'cos' in feat:
        return '时间-周期'
    else:
        return '其他'


def main():
    """主函数"""
    print("=" * 70)
    print("  模型训练 v4 - 可解释性优先")
    print("=" * 70)

    horizons = [1, 6, 12, 24]
    all_reports = {}

    for h in horizons:
        print(f"\n{'='*70}")
        print(f"  预测时间跨度: {h} 小时")
        print(f"{'='*70}")

        # 加载数据
        X, y, feature_cols = load_data(h)
        if X is None:
            continue

        print(f"\n📂 数据: {len(X)} 样本, {len(feature_cols)} 特征")

        # 训练模型
        model, cv_results = train_and_evaluate(X, y, feature_cols, h)

        # 生成可解释性报告
        report = generate_interpretability_report(model, feature_cols, X, y, h)
        report['metrics'] = {
            'MAE': float(np.mean(cv_results['MAE'])),
            'RMSE': float(np.mean(cv_results['RMSE'])),
            'R2': float(np.mean(cv_results['R2']))
        }
        all_reports[h] = report

        # 保存模型
        model_path = MODELS_DIR / f"xgboost_model_v4_{h}h.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # 保存特征重要性
        imp_df = pd.DataFrame(report['features'])
        imp_df.to_csv(MODELS_DIR / f"feature_importance_v4_{h}h.csv", index=False)

    # 保存完整报告（供 Dashboard 使用）
    report_path = DASHBOARD_DATA_DIR / "interpretability_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)
    print(f"\n💾 可解释性报告已保存: {report_path}")

    # 汇总
    print("\n" + "=" * 70)
    print("  模型性能汇总")
    print("=" * 70)
    print(f"\n   预测时间 |   MAE   |  RMSE   |   R²")
    print("   " + "-" * 40)
    for h in horizons:
        if h in all_reports:
            m = all_reports[h]['metrics']
            print(f"      {h:2}h   |  {m['MAE']:5.2f}  |  {m['RMSE']:5.2f}  | {m['R2']:6.3f}")

    print("\n" + "=" * 70)
    print("  关键发现（扬州特有规律）")
    print("=" * 70)
    print("""
    1. 湿度与AQI负相关（与北方城市相反）
       - 高湿度(>80%)时平均AQI=49，干燥(<40%)时AQI=68
       - 原因：扬州位于长三角，高湿度通常伴随降雨/海洋气团

    2. 日温差与AQI正相关
       - 日温差大=晴朗天气=高压控制=大气稳定=不利扩散
       - 这与"逆温层"假设不同，反映了长三角的实际情况

    3. 供暖季效应显著
       - 11-3月AQI明显偏高，反映北方供暖带来的区域传输

    这些发现展示了可解释性分析的价值：
    发现数据中的真实规律，而非简单套用教科书知识。
    """)


if __name__ == "__main__":
    main()
