# -*- coding: utf-8 -*-
"""
模型训练 v3 - 方法论正确版本

为每个预测时间跨度（1h, 6h, 12h, 24h）训练独立的模型
每个模型只使用该时间跨度下合法可知的特征
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


def load_data(horizon_hours):
    """加载特定预测时间跨度的特征数据"""
    input_file = FEATURES_DATA_DIR / f"yangzhou_features_v3_{horizon_hours}h.csv"

    if not input_file.exists():
        print(f"❌ 文件不存在: {input_file}")
        print(f"   请先运行 build_features_v3.py")
        return None, None, None

    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 分离特征和目标
    feature_cols = [c for c in df.columns if c not in ['datetime', 'AQI_target']]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['AQI_target']

    return X, y, feature_cols


def train_baseline(X, y, horizon_hours):
    """训练基线模型"""
    print(f"\n📊 基线模型评估 ({horizon_hours}h 预测)...")

    results = {}

    # Naive: 用 lag_0h（即 horizon_hours 前的 AQI）作为预测
    if 'AQI_lag_0h' in X.columns:
        naive_pred = X['AQI_lag_0h'].values
        valid_idx = ~np.isnan(naive_pred)
        if valid_idx.sum() > 0:
            naive_mae = mean_absolute_error(y[valid_idx], naive_pred[valid_idx])
            naive_r2 = r2_score(y[valid_idx], naive_pred[valid_idx])
            results['Naive (lag_0h)'] = {'MAE': naive_mae, 'R2': naive_r2}
            print(f"   Naive (用 {horizon_hours}h 前的 AQI): MAE={naive_mae:.2f}, R²={naive_r2:.4f}")

    # 历史同期
    if 'AQI_yesterday_same_hour' in X.columns:
        hist_pred = X['AQI_yesterday_same_hour'].values
        valid_idx = ~np.isnan(hist_pred)
        if valid_idx.sum() > 0:
            hist_mae = mean_absolute_error(y[valid_idx], hist_pred[valid_idx])
            hist_r2 = r2_score(y[valid_idx], hist_pred[valid_idx])
            results['Historical (昨天同期)'] = {'MAE': hist_mae, 'R2': hist_r2}
            print(f"   Historical (昨天同期): MAE={hist_mae:.2f}, R²={hist_r2:.4f}")

    return results


def train_xgboost(X, y, feature_cols, horizon_hours):
    """训练 XGBoost 模型"""
    print(f"\n🤖 训练 XGBoost 模型 ({horizon_hours}h 预测)...")

    # 时序交叉验证
    tscv = TimeSeriesSplit(n_splits=5)

    cv_results = {'MAE': [], 'RMSE': [], 'R2': [], 'MAPE': []}

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

        # 安全计算 MAPE
        mask = y_test > 10  # 避免除以接近零的值
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        else:
            mape = np.nan

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
    print(f"      MAPE: {np.nanmean(cv_results['MAPE']):.2f}%")

    # 训练最终模型
    print(f"\n   训练最终模型...")
    final_model = XGBRegressor(**params)
    final_model.fit(X, y, verbose=False)

    return final_model, cv_results


def analyze_feature_importance(model, feature_cols, horizon_hours):
    """分析特征重要性"""
    print(f"\n📊 特征重要性分析 ({horizon_hours}h)...")

    importance = model.feature_importances_
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print("   Top 10 重要特征:")
    for i, row in imp_df.head(10).iterrows():
        pct = row['importance'] * 100
        bar = '█' * int(pct * 2)
        print(f"      {row['feature']:35} {pct:5.2f}% {bar}")

    return imp_df


def main():
    """主函数"""
    print("=" * 60)
    print("  模型训练 v3 - 方法论正确版本")
    print("  为每个预测时间跨度训练独立模型")
    print("=" * 60)

    # 预测时间跨度
    horizons = [1, 6, 12, 24]

    all_results = {}

    for h in horizons:
        print(f"\n{'='*60}")
        print(f"  预测时间跨度: {h} 小时")
        print(f"{'='*60}")

        # 加载数据
        X, y, feature_cols = load_data(h)
        if X is None:
            continue

        print(f"\n📂 数据加载完成:")
        print(f"   样本数: {len(X)}")
        print(f"   特征数: {len(feature_cols)}")

        # 基线模型
        baseline_results = train_baseline(X, y, h)

        # XGBoost
        model, cv_results = train_xgboost(X, y, feature_cols, h)

        # 特征重要性
        imp_df = analyze_feature_importance(model, feature_cols, h)

        # 保存模型
        model_path = MODELS_DIR / f"xgboost_model_v3_{h}h.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n💾 模型已保存: {model_path}")

        # 保存特征重要性
        imp_path = MODELS_DIR / f"feature_importance_v3_{h}h.csv"
        imp_df.to_csv(imp_path, index=False)

        # 记录结果
        all_results[h] = {
            'baseline': baseline_results,
            'xgboost': {
                'MAE': np.mean(cv_results['MAE']),
                'RMSE': np.mean(cv_results['RMSE']),
                'R2': np.mean(cv_results['R2']),
                'MAPE': np.nanmean(cv_results['MAPE'])
            },
            'improvement': None
        }

        # 计算相对基线的提升
        if 'Naive (lag_0h)' in baseline_results:
            baseline_mae = baseline_results['Naive (lag_0h)']['MAE']
            xgb_mae = np.mean(cv_results['MAE'])
            improvement = (baseline_mae - xgb_mae) / baseline_mae * 100
            all_results[h]['improvement'] = improvement

    # 汇总报告
    print("\n" + "=" * 60)
    print("  模型训练汇总")
    print("=" * 60)

    print("\n预测时间 | Naive MAE | XGBoost MAE | R² | 提升")
    print("-" * 60)

    for h in horizons:
        if h not in all_results:
            continue
        r = all_results[h]
        naive_mae = r['baseline'].get('Naive (lag_0h)', {}).get('MAE', np.nan)
        xgb_mae = r['xgboost']['MAE']
        r2 = r['xgboost']['R2']
        imp = r['improvement'] if r['improvement'] else 0

        print(f"   {h:2}h    |   {naive_mae:5.2f}   |    {xgb_mae:5.2f}    | {r2:.3f} | {imp:+5.1f}%")

    print("\n" + "=" * 60)
    print("  关键发现")
    print("=" * 60)
    print("""
    1. 随着预测时间增加，MAE 会上升，R² 会下降 —— 这是合理的
    2. Naive baseline (用历史值预测) 是重要的对比基准
    3. 如果 XGBoost 相比 Naive 提升不大，说明 AQI 主要靠惯性
    4. 这个结果是诚实的：没有用不可知的信息来"作弊"
    """)


if __name__ == "__main__":
    main()
