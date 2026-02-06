# -*- coding: utf-8 -*-
"""
模型解释模块
使用 SHAP 解释模型预测
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import FEATURES_DATA_DIR, MODELS_DIR, DOCS_DIR, DASHBOARD_DATA_DIR

# 尝试导入 SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ SHAP 未安装，模型解释功能将不可用")


def load_model():
    """加载训练好的模型"""
    model_path = MODELS_DIR / "xgboost_model.pkl"

    if not model_path.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return None

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model


def _patch_shap_xgboost_compat():
    """修复 XGBoost 3.x + SHAP 兼容性问题
    XGBoost 3.x 将 base_score 存为 '[5.97E1]' 格式（UBJSON），SHAP float() 无法解析
    通过 monkey-patch XGBTreeModelLoader.__init__ 在 UBJSON 解码后修复"""
    try:
        import shap.explainers._tree as tree_mod
        _orig_init = tree_mod.XGBTreeModelLoader.__init__

        if getattr(tree_mod.XGBTreeModelLoader, '_patched', False):
            return

        # 保存原始 decode_ubjson_buffer
        _orig_decode = tree_mod.decode_ubjson_buffer

        def _fix_bracket_values(obj):
            """递归修复 dict 中 '[xxx]' 格式的字符串值"""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and v.startswith('[') and v.endswith(']'):
                        try:
                            obj[k] = str(float(v.strip('[]')))
                        except ValueError:
                            pass
                    elif isinstance(v, (dict, list)):
                        _fix_bracket_values(v)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        _fix_bracket_values(item)

        def _patched_decode(fd):
            result = _orig_decode(fd)
            _fix_bracket_values(result)
            return result

        tree_mod.decode_ubjson_buffer = _patched_decode
        tree_mod.XGBTreeModelLoader._patched = True
        print("   🔧 已修复 XGBoost 3.x / SHAP 兼容性")
    except Exception as e:
        print(f"   ⚠️ SHAP 兼容性修复失败: {e}")


def compute_shap_values(model, X, max_samples=1000):
    """
    计算 SHAP 值

    Args:
        model: 训练好的模型
        X: 特征数据
        max_samples: 最大样本数（用于加速）

    Returns:
        shap.Explanation: SHAP 值
    """
    if not HAS_SHAP:
        print("❌ SHAP 未安装")
        return None

    print("🔍 计算 SHAP 值...")

    # 修复 XGBoost 3.x + SHAP 兼容性
    _patch_shap_xgboost_compat()

    # 如果样本太多，随机采样
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    # 创建 SHAP 解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    print(f"   完成! 样本数: {len(X_sample)}")

    return shap_values, X_sample


def get_global_feature_importance(shap_values):
    """
    获取全局特征重要性

    Args:
        shap_values: SHAP 值

    Returns:
        DataFrame: 特征重要性排序
    """
    importance = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': shap_values.feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return importance_df


def explain_single_prediction(model, X, index, feature_names):
    """
    解释单个预测

    Args:
        model: 模型
        X: 特征数据
        index: 要解释的样本索引
        feature_names: 特征名列表

    Returns:
        dict: 解释结果
    """
    if not HAS_SHAP:
        return None

    explainer = shap.TreeExplainer(model)

    if isinstance(X, pd.DataFrame):
        sample = X.iloc[[index]]
    else:
        sample = X[index:index+1]

    shap_values = explainer(sample)

    # 获取每个特征的贡献
    contributions = []
    for i, (feat, val, shap_val) in enumerate(zip(feature_names, sample.values[0], shap_values.values[0])):
        contributions.append({
            'feature': feat,
            'value': float(val) if not np.isnan(val) else None,
            'shap_value': float(shap_val)
        })

    # 按 SHAP 值绝对值排序
    contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

    return {
        'prediction': float(model.predict(sample)[0]),
        'base_value': float(shap_values.base_values[0]),
        'contributions': contributions[:20]  # 只返回 Top 20
    }


def generate_shap_summary_data(shap_values, X_sample, top_n=20):
    """
    生成 SHAP Summary Plot 所需的数据

    Args:
        shap_values: SHAP 值
        X_sample: 样本数据
        top_n: 显示的特征数量

    Returns:
        dict: 可用于前端可视化的数据
    """
    # 获取最重要的特征
    importance = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(importance)[-top_n:][::-1]

    feature_names = [shap_values.feature_names[i] for i in top_indices]

    # 准备数据
    summary_data = []
    for i, feat_idx in enumerate(top_indices):
        feat_name = shap_values.feature_names[feat_idx]

        # 获取该特征的 SHAP 值和原始值
        shap_vals = shap_values.values[:, feat_idx]
        feat_vals = X_sample.iloc[:, feat_idx].values if isinstance(X_sample, pd.DataFrame) else X_sample[:, feat_idx]

        # 归一化特征值用于着色
        feat_min = np.nanmin(feat_vals)
        feat_max = np.nanmax(feat_vals)
        if feat_max > feat_min:
            feat_normalized = (feat_vals - feat_min) / (feat_max - feat_min)
        else:
            feat_normalized = np.zeros_like(feat_vals)

        summary_data.append({
            'feature': feat_name,
            'importance': float(importance[feat_idx]),
            'shap_values': shap_vals.tolist(),
            'feature_values': feat_vals.tolist(),
            'feature_normalized': feat_normalized.tolist()
        })

    return {
        'features': feature_names,
        'data': summary_data
    }


def main():
    """主函数"""
    print("=" * 60)
    print("  模型解释 (SHAP)")
    print("=" * 60)

    if not HAS_SHAP:
        print("❌ SHAP 未安装，请运行: pip install shap")
        return

    # 加载模型
    model = load_model()
    if model is None:
        return

    # 读取特征数据
    input_file = FEATURES_DATA_DIR / "yangzhou_features_selected.csv"
    if not input_file.exists():
        input_file = FEATURES_DATA_DIR / "yangzhou_features.csv"

    if not input_file.exists():
        print(f"❌ 特征文件不存在: {input_file}")
        return

    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 准备特征
    feature_cols = [c for c in df.columns if c not in ['datetime', 'AQI', 'AQI_target', 'date']]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

    print(f"📖 读取数据: {len(X)} 行, {len(feature_cols)} 个特征")

    # 计算 SHAP 值
    shap_values, X_sample = compute_shap_values(model, X)

    if shap_values is None:
        return

    # 获取全局特征重要性
    importance_df = get_global_feature_importance(shap_values)
    print("\n📊 Top 15 特征重要性 (SHAP):")
    for i, row in importance_df.head(15).iterrows():
        print(f"   {i+1:2d}. {row['feature']:40s} {row['importance']:.4f}")

    # 保存 SHAP 值
    shap_file = MODELS_DIR / "shap_values.pkl"
    with open(shap_file, 'wb') as f:
        pickle.dump({
            'shap_values': shap_values,
            'X_sample': X_sample,
            'feature_names': feature_cols
        }, f)
    print(f"\n✅ SHAP 值已保存: {shap_file}")

    # 生成 Dashboard 数据
    summary_data = generate_shap_summary_data(shap_values, X_sample)

    dashboard_file = DASHBOARD_DATA_DIR / "shap_summary.json"
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False)

    print(f"✅ Dashboard SHAP 数据已保存: {dashboard_file}")

    # 保存特征重要性（SHAP版本）
    importance_file = DOCS_DIR / "shap_feature_importance.json"
    importance_df.to_json(importance_file, orient='records', force_ascii=False, indent=2)
    print(f"✅ SHAP 特征重要性已保存: {importance_file}")

    return shap_values


if __name__ == "__main__":
    main()
