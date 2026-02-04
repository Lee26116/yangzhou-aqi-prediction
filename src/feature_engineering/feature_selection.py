# -*- coding: utf-8 -*-
"""
特征筛选模块
通过相关性分析、VIF、特征重要性等方法筛选特征
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FEATURES_DATA_DIR, DOCS_DIR


def calculate_correlation(df, target_col='AQI'):
    """
    计算特征与目标变量的相关性

    Args:
        df: DataFrame
        target_col: 目标变量列名

    Returns:
        DataFrame: 相关性结果
    """
    print("📊 计算特征相关性...")

    # 只选择数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if target_col not in numeric_cols:
        print(f"   ❌ 目标变量 {target_col} 不是数值类型")
        return None

    # 计算相关系数
    correlations = []
    for col in numeric_cols:
        if col == target_col:
            continue

        try:
            corr = df[col].corr(df[target_col])
            correlations.append({
                'feature': col,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })
        except:
            continue

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)

    return corr_df


def calculate_vif(df, features):
    """
    计算方差膨胀因子 (VIF)

    Args:
        df: DataFrame
        features: 特征列表

    Returns:
        DataFrame: VIF 结果
    """
    print("📊 计算 VIF...")

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # 准备数据
    X = df[features].dropna()

    if len(X) == 0:
        print("   ❌ 数据为空")
        return None

    # 计算 VIF
    vif_data = []
    for i, col in enumerate(features):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({
                'feature': col,
                'VIF': vif
            })
        except Exception as e:
            vif_data.append({
                'feature': col,
                'VIF': np.nan
            })

    vif_df = pd.DataFrame(vif_data)
    vif_df = vif_df.sort_values('VIF', ascending=False)

    return vif_df


def calculate_feature_importance(df, target_col='AQI', n_estimators=100):
    """
    使用随机森林计算特征重要性

    Args:
        df: DataFrame
        target_col: 目标变量列名
        n_estimators: 树的数量

    Returns:
        DataFrame: 特征重要性结果
    """
    print("📊 计算特征重要性 (随机森林)...")

    # 准备数据
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]

    X = df[feature_cols].dropna()
    y = df.loc[X.index, target_col]

    # 处理无穷值
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_idx = ~X.isna().any(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    if len(X) < 100:
        print(f"   ❌ 样本数太少: {len(X)}")
        return None

    # 训练随机森林
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # 获取特征重要性
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False)

    return importance_df


def calculate_mutual_information(df, target_col='AQI'):
    """
    计算互信息

    Args:
        df: DataFrame
        target_col: 目标变量列名

    Returns:
        DataFrame: 互信息结果
    """
    print("📊 计算互信息...")

    # 准备数据
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 处理缺失值和无穷值
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_idx = ~X.isna().any(axis=1) & ~y.isna()
    X = X[valid_idx].values
    y = y[valid_idx].values

    if len(X) < 100:
        print(f"   ❌ 样本数太少: {len(X)}")
        return None

    # 计算互信息
    mi = mutual_info_regression(X, y, random_state=42)

    mi_df = pd.DataFrame({
        'feature': feature_cols,
        'mutual_info': mi
    })
    mi_df = mi_df.sort_values('mutual_info', ascending=False)

    return mi_df


def select_features(df, target_col='AQI',
                   min_correlation=0.1,
                   max_vif=10,
                   max_missing_rate=0.2,
                   top_n_importance=50):
    """
    综合特征筛选

    Args:
        df: DataFrame
        target_col: 目标变量列名
        min_correlation: 最小相关系数绝对值
        max_vif: 最大 VIF 值
        max_missing_rate: 最大缺失率
        top_n_importance: 保留的特征数量

    Returns:
        tuple: (selected_features, excluded_features, selection_report)
    """
    print("\n🔍 开始特征筛选...")

    # 准备结果
    excluded_features = []
    selection_reasons = {}

    # 获取所有数值特征
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_col]

    print(f"   初始特征数: {len(feature_cols)}")

    # 1. 缺失率筛选
    print("\n   [1/4] 缺失率筛选...")
    missing_rates = df[feature_cols].isna().mean()
    high_missing = missing_rates[missing_rates > max_missing_rate].index.tolist()

    for col in high_missing:
        excluded_features.append(col)
        selection_reasons[col] = f"缺失率过高: {missing_rates[col]*100:.1f}% > {max_missing_rate*100:.0f}%"

    remaining_features = [c for c in feature_cols if c not in excluded_features]
    print(f"      排除 {len(high_missing)} 个高缺失特征，剩余 {len(remaining_features)} 个")

    # 2. 相关性筛选
    print("\n   [2/4] 相关性筛选...")
    corr_df = calculate_correlation(df, target_col)

    if corr_df is not None:
        low_corr = corr_df[corr_df['abs_correlation'] < min_correlation]['feature'].tolist()
        for col in low_corr:
            if col not in excluded_features:
                excluded_features.append(col)
                corr_val = corr_df[corr_df['feature'] == col]['correlation'].values[0]
                selection_reasons[col] = f"相关性过低: {corr_val:.3f}"

        remaining_features = [c for c in feature_cols if c not in excluded_features]
        print(f"      排除 {len(low_corr)} 个低相关特征，剩余 {len(remaining_features)} 个")

    # 3. 特征重要性排序
    print("\n   [3/4] 特征重要性排序...")
    importance_df = calculate_feature_importance(df[remaining_features + [target_col]], target_col)

    if importance_df is not None and len(importance_df) > top_n_importance:
        low_importance = importance_df.iloc[top_n_importance:]['feature'].tolist()
        for col in low_importance:
            if col not in excluded_features:
                excluded_features.append(col)
                imp_val = importance_df[importance_df['feature'] == col]['importance'].values[0]
                selection_reasons[col] = f"重要性较低: {imp_val:.4f}"

        remaining_features = [c for c in remaining_features if c not in low_importance]
        print(f"      保留 Top {top_n_importance} 重要特征，剩余 {len(remaining_features)} 个")

    # 4. VIF 筛选（处理多重共线性）
    print("\n   [4/4] VIF 筛选...")
    # VIF 计算较慢，只对剩余特征计算
    if len(remaining_features) > 5:
        try:
            vif_df = calculate_vif(df[remaining_features].dropna(), remaining_features)

            if vif_df is not None:
                # 迭代删除高 VIF 特征
                while True:
                    high_vif = vif_df[vif_df['VIF'] > max_vif]
                    if len(high_vif) == 0:
                        break

                    # 删除 VIF 最高的特征
                    worst_feature = high_vif.iloc[0]['feature']
                    if worst_feature not in excluded_features:
                        excluded_features.append(worst_feature)
                        selection_reasons[worst_feature] = f"VIF 过高: {high_vif.iloc[0]['VIF']:.1f} > {max_vif}"

                    remaining_features = [c for c in remaining_features if c != worst_feature]

                    if len(remaining_features) < 5:
                        break

                    # 重新计算 VIF
                    vif_df = calculate_vif(df[remaining_features].dropna(), remaining_features)
                    if vif_df is None:
                        break

                print(f"      VIF 筛选后剩余 {len(remaining_features)} 个特征")
        except Exception as e:
            print(f"      ⚠️ VIF 计算失败: {e}")

    # 最终结果
    selected_features = remaining_features

    print(f"\n✅ 特征筛选完成!")
    print(f"   最终保留特征: {len(selected_features)} 个")
    print(f"   排除特征: {len(excluded_features)} 个")

    # 生成筛选报告
    selection_report = {
        "initial_features": len(feature_cols),
        "selected_features": len(selected_features),
        "excluded_features": len(excluded_features),
        "parameters": {
            "min_correlation": min_correlation,
            "max_vif": max_vif,
            "max_missing_rate": max_missing_rate,
            "top_n_importance": top_n_importance
        }
    }

    return selected_features, excluded_features, selection_reasons, selection_report


def save_selection_results(selected_features, excluded_features, selection_reasons, corr_df, importance_df):
    """保存特征筛选结果"""

    # 1. 保存被排除的特征
    excluded_file = DOCS_DIR / "feature_excluded.json"
    excluded_data = []
    for feat in excluded_features:
        excluded_data.append({
            "feature": feat,
            "reason": selection_reasons.get(feat, "未知")
        })

    with open(excluded_file, 'w', encoding='utf-8') as f:
        json.dump(excluded_data, f, ensure_ascii=False, indent=2)

    print(f"📄 排除特征列表已保存: {excluded_file}")

    # 2. 保存选中的特征
    selected_file = DOCS_DIR / "feature_selected.json"

    selected_data = []
    for feat in selected_features:
        item = {"feature": feat}

        # 添加相关性
        if corr_df is not None:
            corr_row = corr_df[corr_df['feature'] == feat]
            if len(corr_row) > 0:
                item["correlation"] = round(corr_row['correlation'].values[0], 4)

        # 添加重要性
        if importance_df is not None:
            imp_row = importance_df[importance_df['feature'] == feat]
            if len(imp_row) > 0:
                item["importance"] = round(imp_row['importance'].values[0], 4)

        selected_data.append(item)

    # 按重要性排序
    selected_data.sort(key=lambda x: x.get('importance', 0), reverse=True)

    with open(selected_file, 'w', encoding='utf-8') as f:
        json.dump(selected_data, f, ensure_ascii=False, indent=2)

    print(f"📄 选中特征列表已保存: {selected_file}")


def main():
    """主函数"""
    print("=" * 60)
    print("  特征筛选")
    print("=" * 60)

    # 读取特征数据
    input_file = FEATURES_DATA_DIR / "yangzhou_features.csv"

    if not input_file.exists():
        print(f"❌ 特征文件不存在: {input_file}")
        print("   请先运行 build_features.py")
        return

    df = pd.read_csv(input_file)
    print(f"📖 读取数据: {len(df)} 行, {len(df.columns)} 列")

    # 计算相关性
    corr_df = calculate_correlation(df)

    # 计算特征重要性
    importance_df = calculate_feature_importance(df)

    # 特征筛选
    selected, excluded, reasons, report = select_features(df)

    # 保存结果
    save_selection_results(selected, excluded, reasons, corr_df, importance_df)

    # 保存筛选后的数据
    output_cols = ['datetime', 'AQI'] + selected
    output_cols = [c for c in output_cols if c in df.columns]

    df_selected = df[output_cols]
    output_file = FEATURES_DATA_DIR / "yangzhou_features_selected.csv"
    df_selected.to_csv(output_file, index=False, encoding='utf-8')

    print(f"\n✅ 筛选后的特征数据已保存: {output_file}")
    print(f"   保留特征: {len(selected)} 个")

    # 打印 Top 20 特征
    print("\n📊 Top 20 重要特征:")
    if importance_df is not None:
        for i, row in importance_df.head(20).iterrows():
            print(f"   {i+1:2d}. {row['feature']:40s} {row['importance']:.4f}")

    return selected


if __name__ == "__main__":
    main()
