# -*- coding: utf-8 -*-
"""
模型训练模块
包括基线模型和主模型训练
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FEATURES_DATA_DIR, MODELS_DIR, DOCS_DIR

# 尝试导入 XGBoost 和 LightGBM
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost 未安装，将使用 GradientBoosting 代替")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def calculate_metrics(y_true, y_pred):
    """
    计算评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        dict: 评估指标
    """
    # 过滤无效值
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (避免除以零)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 4),
        'MAPE': round(mape, 2) if not np.isnan(mape) else None
    }


class BaselineModels:
    """基线模型"""

    @staticmethod
    def naive_yesterday(y, hours_lag=24):
        """
        天真预测：使用昨天同一时刻的值

        Args:
            y: 目标序列
            hours_lag: 滞后小时数

        Returns:
            预测值
        """
        return y.shift(hours_lag)

    @staticmethod
    def moving_average(y, window=24):
        """
        移动平均预测

        Args:
            y: 目标序列
            window: 窗口大小

        Returns:
            预测值
        """
        return y.rolling(window=window, min_periods=1).mean().shift(1)

    @staticmethod
    def exponential_smoothing(y, alpha=0.3):
        """
        指数平滑预测

        Args:
            y: 目标序列
            alpha: 平滑系数

        Returns:
            预测值
        """
        result = y.copy()
        for i in range(1, len(y)):
            result.iloc[i] = alpha * y.iloc[i-1] + (1 - alpha) * result.iloc[i-1]
        return result.shift(1)


def train_baseline_models(df, target_col='AQI'):
    """
    训练和评估基线模型

    Args:
        df: DataFrame
        target_col: 目标变量列名

    Returns:
        dict: 基线模型结果
    """
    print("\n📊 训练基线模型...")

    y = df[target_col].copy()

    results = {}

    # 1. 天真预测（昨天同一时刻）
    pred_naive = BaselineModels.naive_yesterday(y, 24)
    valid_idx = ~(pred_naive.isna() | y.isna())
    results['Naive_24h'] = calculate_metrics(y[valid_idx].values, pred_naive[valid_idx].values)
    print(f"   Naive (24h): MAE={results['Naive_24h']['MAE']}, RMSE={results['Naive_24h']['RMSE']}")

    # 2. 移动平均
    pred_ma = BaselineModels.moving_average(y, 24)
    valid_idx = ~(pred_ma.isna() | y.isna())
    results['MovingAvg_24h'] = calculate_metrics(y[valid_idx].values, pred_ma[valid_idx].values)
    print(f"   Moving Avg (24h): MAE={results['MovingAvg_24h']['MAE']}, RMSE={results['MovingAvg_24h']['RMSE']}")

    # 3. 指数平滑
    pred_es = BaselineModels.exponential_smoothing(y, 0.3)
    valid_idx = ~(pred_es.isna() | y.isna())
    results['ExpSmoothing'] = calculate_metrics(y[valid_idx].values, pred_es[valid_idx].values)
    print(f"   Exp Smoothing: MAE={results['ExpSmoothing']['MAE']}, RMSE={results['ExpSmoothing']['RMSE']}")

    # 4. 单特征线性回归（使用 1 小时滞后的 AQI）
    if 'AQI_lag_1h' in df.columns:
        X_simple = df[['AQI_lag_1h']].dropna()
        y_simple = df.loc[X_simple.index, target_col]

        # 简单训练测试分割（80/20）
        split_idx = int(len(X_simple) * 0.8)
        X_train, X_test = X_simple.iloc[:split_idx], X_simple.iloc[split_idx:]
        y_train, y_test = y_simple.iloc[:split_idx], y_simple.iloc[split_idx:]

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        results['Linear_Lag1h'] = calculate_metrics(y_test.values, pred_lr)
        print(f"   Linear (lag 1h): MAE={results['Linear_Lag1h']['MAE']}, RMSE={results['Linear_Lag1h']['RMSE']}")

    return results


def train_main_model(df, target_col='AQI', feature_cols=None, n_splits=5):
    """
    训练主模型

    Args:
        df: DataFrame
        target_col: 目标变量列名
        feature_cols: 特征列名列表
        n_splits: 时间序列交叉验证折数

    Returns:
        tuple: (model, metrics, feature_importance)
    """
    print("\n🚀 训练主模型...")

    # 准备特征
    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col and c != 'datetime']

    # 删除含有无穷值和缺失值的行
    df_clean = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    print(f"   特征数: {len(feature_cols)}")
    print(f"   样本数: {len(X)}")

    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # 选择模型
    if HAS_XGBOOST:
        print("   使用 XGBoost 模型")
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    else:
        print("   使用 GradientBoosting 模型")
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )

    # 交叉验证
    cv_results = []
    all_predictions = []
    all_actuals = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        metrics = calculate_metrics(y_val.values, y_pred)
        cv_results.append(metrics)
        all_predictions.extend(y_pred)
        all_actuals.extend(y_val.values)

        print(f"   Fold {fold+1}: MAE={metrics['MAE']}, RMSE={metrics['RMSE']}, R²={metrics['R2']}")

    # 计算平均指标
    avg_metrics = {
        'MAE': round(np.mean([r['MAE'] for r in cv_results]), 2),
        'RMSE': round(np.mean([r['RMSE'] for r in cv_results]), 2),
        'R2': round(np.mean([r['R2'] for r in cv_results]), 4),
        'MAPE': round(np.mean([r['MAPE'] for r in cv_results if r['MAPE'] is not None]), 2)
    }

    print(f"\n   平均指标: MAE={avg_metrics['MAE']}, RMSE={avg_metrics['RMSE']}, R²={avg_metrics['R2']}")

    # 在全部数据上重新训练最终模型
    model.fit(X, y)

    # 获取特征重要性
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        importance_df = None

    return model, avg_metrics, importance_df, cv_results


def train_ablation_models(df, target_col='AQI', feature_cols=None):
    """
    消融实验：评估不同特征组的贡献

    Args:
        df: DataFrame
        target_col: 目标变量列名
        feature_cols: 全部特征列名列表

    Returns:
        dict: 消融实验结果
    """
    print("\n🔬 进行消融实验...")

    if feature_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]

    # 特征分组
    feature_groups = {
        '时间特征': [c for c in feature_cols if any(x in c for x in ['hour', 'day', 'month', 'week', 'season', 'sin', 'cos', 'period', 'weekend', 'holiday', 'workday'])],
        '滞后特征': [c for c in feature_cols if 'lag' in c],
        '滚动特征': [c for c in feature_cols if 'rolling' in c],
        '气象特征': [c for c in feature_cols if any(x in c for x in ['temp', 'humidity', 'wind', 'pressure', 'precipitation', 'cloud', 'rain'])],
        '变化率特征': [c for c in feature_cols if 'change' in c or 'pct' in c],
        '代理变量': [c for c in feature_cols if 'proxy' in c or 'inversion' in c or 'burning' in c or 'festival' in c or 'heating' in c],
    }

    # 删除空组
    feature_groups = {k: v for k, v in feature_groups.items() if len(v) > 0}

    results = {}

    # 准备数据
    df_clean = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    X = df_clean[feature_cols]
    y = df_clean[target_col]

    # 1. 完整模型
    print("   训练完整模型...")
    full_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    tscv = TimeSeriesSplit(n_splits=3)

    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        full_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = full_model.predict(X.iloc[val_idx])
        cv_scores.append(calculate_metrics(y.iloc[val_idx].values, pred))

    results['完整模型'] = {
        'features_used': len(feature_cols),
        'MAE': round(np.mean([s['MAE'] for s in cv_scores]), 2),
        'RMSE': round(np.mean([s['RMSE'] for s in cv_scores]), 2),
        'R2': round(np.mean([s['R2'] for s in cv_scores]), 4)
    }

    # 2. 移除每组特征后的模型
    for group_name, group_features in feature_groups.items():
        remaining_features = [c for c in feature_cols if c not in group_features]

        if len(remaining_features) == 0:
            continue

        print(f"   训练去除 [{group_name}] 后的模型...")

        X_ablation = df_clean[remaining_features]

        cv_scores = []
        for train_idx, val_idx in tscv.split(X_ablation):
            model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_ablation.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X_ablation.iloc[val_idx])
            cv_scores.append(calculate_metrics(y.iloc[val_idx].values, pred))

        results[f'去除{group_name}'] = {
            'features_removed': len(group_features),
            'features_used': len(remaining_features),
            'MAE': round(np.mean([s['MAE'] for s in cv_scores]), 2),
            'RMSE': round(np.mean([s['RMSE'] for s in cv_scores]), 2),
            'R2': round(np.mean([s['R2'] for s in cv_scores]), 4)
        }

        # 计算该组特征的贡献
        mae_increase = results[f'去除{group_name}']['MAE'] - results['完整模型']['MAE']
        results[f'去除{group_name}']['MAE_increase'] = round(mae_increase, 2)

    return results


def save_model(model, model_name='xgboost_model'):
    """保存模型"""
    # 保存为 pickle
    pkl_path = MODELS_DIR / f"{model_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)

    # 如果是 XGBoost，也保存为 JSON
    if HAS_XGBOOST and isinstance(model, xgb.XGBRegressor):
        json_path = MODELS_DIR / f"{model_name}.json"
        model.save_model(str(json_path))

    print(f"✅ 模型已保存: {pkl_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("  模型训练")
    print("=" * 60)

    # 读取特征数据
    input_file = FEATURES_DATA_DIR / "yangzhou_features_selected.csv"

    if not input_file.exists():
        # 尝试使用完整特征文件
        input_file = FEATURES_DATA_DIR / "yangzhou_features.csv"

    if not input_file.exists():
        print(f"❌ 特征文件不存在: {input_file}")
        return

    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"📖 读取数据: {len(df)} 行, {len(df.columns)} 列")

    # 获取特征列
    feature_cols = [c for c in df.columns if c not in ['datetime', 'AQI', 'date']]

    # 1. 训练基线模型
    baseline_results = train_baseline_models(df)

    # 2. 训练主模型
    model, main_metrics, importance_df, cv_results = train_main_model(df, feature_cols=feature_cols)

    # 3. 消融实验
    ablation_results = train_ablation_models(df, feature_cols=feature_cols)

    # 保存模型
    save_model(model)

    # 保存评估结果
    evaluation = {
        'timestamp': datetime.now().isoformat(),
        'data_rows': len(df),
        'feature_count': len(feature_cols),
        'baseline_results': baseline_results,
        'main_model': {
            'type': 'XGBoost' if HAS_XGBOOST else 'GradientBoosting',
            'cv_folds': 5,
            'metrics': main_metrics,
            'cv_results': cv_results
        },
        'ablation_study': ablation_results
    }

    eval_file = DOCS_DIR / "model_evaluation.json"
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    print(f"\n📄 评估结果已保存: {eval_file}")

    # 保存基线对比
    baseline_file = DOCS_DIR / "baseline_comparison.json"
    comparison = {
        'baseline_models': baseline_results,
        'main_model': main_metrics,
        'improvement': {
            'vs_naive': round(baseline_results.get('Naive_24h', {}).get('MAE', 0) - main_metrics['MAE'], 2),
            'vs_moving_avg': round(baseline_results.get('MovingAvg_24h', {}).get('MAE', 0) - main_metrics['MAE'], 2),
        }
    }
    with open(baseline_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"📄 基线对比已保存: {baseline_file}")

    # 保存消融实验结果
    ablation_file = DOCS_DIR / "ablation_study.json"
    with open(ablation_file, 'w', encoding='utf-8') as f:
        json.dump(ablation_results, f, ensure_ascii=False, indent=2)

    print(f"📄 消融实验结果已保存: {ablation_file}")

    # 保存特征重要性
    if importance_df is not None:
        importance_file = DOCS_DIR / "feature_importance.json"
        importance_df.to_json(importance_file, orient='records', force_ascii=False, indent=2)
        print(f"📄 特征重要性已保存: {importance_file}")

    print("\n✅ 模型训练完成!")

    return model, evaluation


if __name__ == "__main__":
    main()
