#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扬州空气质量预测系统 - 数据处理管道
一键运行完整的数据处理和模型训练流程
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def run_step(step_name, module_path):
    """运行单个步骤"""
    print("\n" + "=" * 60)
    print(f"  {step_name}")
    print("=" * 60)

    start_time = time.time()

    try:
        # 动态导入并运行
        import importlib.util
        spec = importlib.util.spec_from_file_location("module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'main'):
            module.main()

        elapsed = time.time() - start_time
        print(f"\n✅ {step_name} 完成! 耗时: {elapsed:.1f} 秒")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {step_name} 失败: {e}")
        print(f"   耗时: {elapsed:.1f} 秒")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  扬州空气质量预测系统 - 数据处理管道")
    print("=" * 60)

    project_root = Path(__file__).parent

    # 定义处理步骤
    steps = [
        ("步骤 1: 下载历史空气质量数据", project_root / "src/data_collection/fetch_historical.py"),
        ("步骤 2: 下载历史天气数据", project_root / "src/data_collection/fetch_openmeteo.py"),
        ("步骤 3: 获取节假日数据", project_root / "src/data_collection/fetch_holidays.py"),
        ("步骤 4: 数据清洗", project_root / "src/data_processing/clean_data.py"),
        ("步骤 5: 数据合并", project_root / "src/data_processing/merge_data.py"),
        ("步骤 6: 特征工程", project_root / "src/feature_engineering/build_features.py"),
        ("步骤 7: 特征筛选", project_root / "src/feature_engineering/feature_selection.py"),
        ("步骤 8: 模型训练", project_root / "src/modeling/train_model.py"),
    ]

    # 可选步骤
    optional_steps = [
        ("步骤 9: SHAP 模型解释", project_root / "src/modeling/explain_model.py"),
        ("步骤 10: 生成预测数据", project_root / "src/modeling/predict.py"),
    ]

    # 深度学习步骤
    dl_steps = [
        ("DL 步骤 1: 特征工程", project_root / "deep_learning/data/build_dl_features.py"),
        ("DL 步骤 2: 序列生成", project_root / "deep_learning/data/sequence_builder.py"),
        ("DL 步骤 3: 模型训练", project_root / "deep_learning/training/train.py"),
        ("DL 步骤 4: 模型评估", project_root / "deep_learning/evaluation/evaluate.py"),
        ("DL 步骤 5: ONNX 导出", project_root / "deep_learning/export/export_onnx.py"),
        ("DL 步骤 6: Dashboard 数据", project_root / "deep_learning/dashboard/generate_dashboard_data.py"),
        ("DL 步骤 7: 性能基准", project_root / "deep_learning/benchmark.py"),
    ]

    total_start = time.time()
    results = []

    # 运行必要步骤
    for step_name, module_path in steps:
        success = run_step(step_name, module_path)
        results.append((step_name, success))

        if not success:
            print(f"\n⚠️ {step_name} 失败，停止后续步骤")
            break

    # 运行可选步骤
    if all(r[1] for r in results):
        for step_name, module_path in optional_steps:
            success = run_step(step_name, module_path)
            results.append((step_name, success))

    # 运行深度学习步骤（仅需必要步骤1-8成功，可选步骤失败不阻塞）
    required_ok = all(r[1] for r in results[:len(steps)])
    if required_ok:
        print("\n" + "=" * 60)
        print("  深度学习流水线")
        print("=" * 60)
        for step_name, module_path in dl_steps:
            if module_path.exists():
                success = run_step(step_name, module_path)
                results.append((step_name, success))
                if not success:
                    print(f"\n⚠️ {step_name} 失败，停止后续DL步骤")
                    break
            else:
                print(f"\n⚠️ 跳过 {step_name}: 文件不存在")

    # 总结
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("  处理完成!")
    print("=" * 60)
    print(f"\n总耗时: {total_elapsed/60:.1f} 分钟")

    print("\n步骤结果:")
    for step_name, success in results:
        status = "✅" if success else "❌"
        print(f"   {status} {step_name}")

    success_count = sum(1 for _, s in results if s)
    print(f"\n成功: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
