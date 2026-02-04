# -*- coding: utf-8 -*-
"""
模型训练 v2 - 使用新特征（不含其他城市同期AQI）
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FEATURES_DATA_DIR, MODELS_DIR


def load_data():
    """加载特征数据"""
    input_file = FEATURES_DATA_DIR / "yangzhou_features_v2_selected.csv"

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        return None, None, None

    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 分离特征和目标
    feature_cols = [c for c in df.columns if c not in ['datetime', 'AQI']]
    X = df[feature_cols]
    y = df['AQI']

    # 处理缺失值
    X = X.fillna(X.median())

    return X, y, feature_cols


def train_baselines(X, y):
    """训练基线模型"""
    print("\n📊 基线模型评估...")

    results = {}

    # 1. Naive: 用1小时前的AQI预测
    if 'AQI_lag_1h' in X.columns:
        naive_pred = X['AQI_lag_1h'].values
        valid_idx = ~np.isnan(naive_pred)
        naive_mae = mean_absolute_error(y[valid_idx], naive_pred[valid_idx])
        naive_r2 = r2_score(y[valid_idx], naive_pred[valid_idx])
        results['Naive (lag 1h)'] = {'MAE': naive_mae, 'R2': naive_r2}
        print(f"   Naive (lag 1h): MAE={naive_mae:.2f}, R²={naive_r2:.4f}")

    # 2. Moving Average: 用过去6小时均值预测
    if 'AQI_rolling_mean_6h' in X.columns:
        ma_pred = X['AQI_rolling_mean_6h'].values
        valid_idx = ~np.isnan(ma_pred)
        ma_mae = mean_absolute_error(y[valid_idx], ma_pred[valid_idx])
        ma_r2 = r2_score(y[valid_idx], ma_pred[valid_idx])
        results['Moving Avg (6h)'] = {'MAE': ma_mae, 'R2': ma_r2}
        print(f"   Moving Avg (6h): MAE={ma_mae:.2f}, R²={ma_r2:.4f}")

    return results


def train_xgboost(X, y, feature_cols):
    """训练 XGBoost 模型"""
    print("\n🤖 训练 XGBoost 模型...")

    # 时序交叉验证
    tscv = TimeSeriesSplit(n_splits=5)

    cv_results = {
        'MAE': [], 'RMSE': [], 'R2': [], 'MAPE': []
    }

    # XGBoost 参数
    params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }

    print(f"   5-fold 时序交叉验证...")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        cv_results['MAE'].append(mae)
        cv_results['RMSE'].append(rmse)
        cv_results['R2'].append(r2)
        cv_results['MAPE'].append(mape)

        print(f"      Fold {fold+1}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")

    # 打印平均结果
    print(f"\n   📈 交叉验证平均结果:")
    print(f"      MAE:  {np.mean(cv_results['MAE']):.2f} ± {np.std(cv_results['MAE']):.2f}")
    print(f"      RMSE: {np.mean(cv_results['RMSE']):.2f} ± {np.std(cv_results['RMSE']):.2f}")
    print(f"      R²:   {np.mean(cv_results['R2']):.4f} ± {np.std(cv_results['R2']):.4f}")
    print(f"      MAPE: {np.mean(cv_results['MAPE']):.2f}% ± {np.std(cv_results['MAPE']):.2f}%")

    # 训练最终模型（使用全部数据）
    print(f"\n   训练最终模型...")
    final_model = XGBRegressor(**params)
    final_model.fit(X, y, verbose=False)

    return final_model, cv_results


def analyze_feature_importance(model, feature_cols):
    """分析特征重要性"""
    print("\n📊 特征重要性分析...")

    importance = model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("   Top 15 重要特征:")
    for i, row in imp_df.head(15).iterrows():
        pct = row['importance'] * 100
        bar = '█' * int(pct * 2)
        print(f"      {row['feature']:30} {pct:5.2f}% {bar}")

    return imp_df


def main():
    """主函数"""
    print("=" * 60)
    print("  模型训练 v2 - 不使用其他城市同期AQI")
    print("=" * 60)

    # 加载数据
    X, y, feature_cols = load_data()
    if X is None:
        return

    print(f"\n📂 数据加载完成:")
    print(f"   样本数: {len(X)}")
    print(f"   特征数: {len(feature_cols)}")

    # 基线模型
    baseline_results = train_baselines(X, y)

    # XGBoost
    model, cv_results = train_xgboost(X, y, feature_cols)

    # 特征重要性
    imp_df = analyze_feature_importance(model, feature_cols)

    # 保存模型
    model_path = MODELS_DIR / "xgboost_model_v2.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 模型已保存: {model_path}")

    # 保存特征重要性
    imp_path = MODELS_DIR / "feature_importance_v2.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"💾 特征重要性已保存: {imp_path}")

    # 打印对比
    print("\n" + "=" * 60)
    print("  模型对比总结")
    print("=" * 60)

    print(f"\n基线模型:")
    for name, metrics in baseline_results.items():
        print(f"   {name}: MAE={metrics['MAE']:.2f}, R²={metrics['R2']:.4f}")

    print(f"\nXGBoost (v2, 无其他城市AQI):")
    print(f"   MAE={np.mean(cv_results['MAE']):.2f}, R²={np.mean(cv_results['R2']):.4f}")

    # 计算相对基线的提升
    if 'Naive (lag 1h)' in baseline_results:
        baseline_mae = baseline_results['Naive (lag 1h)']['MAE']
        xgb_mae = np.mean(cv_results['MAE'])
        improvement = (baseline_mae - xgb_mae) / baseline_mae * 100
        print(f"\n   相对 Naive 基线提升: {improvement:.1f}%")

    return model, cv_results, imp_df


if __name__ == "__main__":
    main()
