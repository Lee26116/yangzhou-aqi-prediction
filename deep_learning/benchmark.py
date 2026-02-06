# -*- coding: utf-8 -*-
"""
性能基准测试
验证: 推理 < 1秒, 内存 < 1GB, 模型文件 < 50MB
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from deep_learning.dl_config import (
    DL_EXPORTED_DIR, DL_SEQUENCES_DIR, DL_MODELS_DIR,
    ONNX_CONFIG, SEQUENCE_LENGTH, BENCHMARK_THRESHOLDS
)


def benchmark_inference_time(n_iterations=100):
    """测试推理速度"""
    print("⏱️ 推理速度测试...")

    onnx_path = DL_EXPORTED_DIR / ONNX_CONFIG['model_filename']
    if not onnx_path.exists():
        print("   ⚠️ ONNX 模型不存在，跳过")
        return None

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        n_features = input_shape[2] if len(input_shape) > 2 else 51

        # 预热
        dummy = np.random.randn(1, SEQUENCE_LENGTH, n_features).astype(np.float32)
        for _ in range(10):
            session.run(None, {input_name: dummy})

        # 计时
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            session.run(None, {input_name: dummy})
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        max_time = np.max(times)

        passed = avg_time < BENCHMARK_THRESHOLDS['inference_time_seconds']
        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"   平均: {avg_time*1000:.2f}ms")
        print(f"   P95:  {p95_time*1000:.2f}ms")
        print(f"   最大: {max_time*1000:.2f}ms")
        print(f"   阈值: <{BENCHMARK_THRESHOLDS['inference_time_seconds']*1000}ms")
        print(f"   {status}")

        return {
            'avg_ms': round(avg_time * 1000, 2),
            'p95_ms': round(p95_time * 1000, 2),
            'max_ms': round(max_time * 1000, 2),
            'passed': passed,
        }

    except ImportError:
        print("   ⚠️ onnxruntime 未安装")
        return None


def benchmark_memory():
    """测试内存占用"""
    print("\n💾 内存占用测试...")

    try:
        import tracemalloc
        tracemalloc.start()

        import onnxruntime as ort
        onnx_path = DL_EXPORTED_DIR / ONNX_CONFIG['model_filename']
        if not onnx_path.exists():
            print("   ⚠️ ONNX 模型不存在")
            return None

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        n_features = input_shape[2] if len(input_shape) > 2 else 51

        dummy = np.random.randn(1, SEQUENCE_LENGTH, n_features).astype(np.float32)
        session.run(None, {input_name: dummy})

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        passed = peak_mb < BENCHMARK_THRESHOLDS['memory_mb']
        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"   当前: {current/1024/1024:.1f} MB")
        print(f"   峰值: {peak_mb:.1f} MB")
        print(f"   阈值: <{BENCHMARK_THRESHOLDS['memory_mb']} MB")
        print(f"   {status}")

        return {
            'current_mb': round(current / 1024 / 1024, 1),
            'peak_mb': round(peak_mb, 1),
            'passed': passed,
        }

    except ImportError:
        print("   ⚠️ 依赖未安装")
        return None


def benchmark_model_size():
    """测试模型文件大小"""
    print("\n📦 模型文件大小测试...")

    onnx_path = DL_EXPORTED_DIR / ONNX_CONFIG['model_filename']
    pt_path = DL_MODELS_DIR / "best_model.pt"

    results = {}

    for name, path in [('ONNX', onnx_path), ('PyTorch', pt_path)]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            passed = size_mb < BENCHMARK_THRESHOLDS['model_size_mb']
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {name}: {size_mb:.2f} MB {status}")
            results[name.lower()] = {
                'size_mb': round(size_mb, 2),
                'passed': passed,
            }
        else:
            print(f"   {name}: 文件不存在")

    return results


def run_benchmark():
    """运行所有基准测试"""
    print("=" * 60)
    print("  性能基准测试")
    print("=" * 60)

    results = {
        'inference': benchmark_inference_time(),
        'memory': benchmark_memory(),
        'model_size': benchmark_model_size(),
    }

    # 总结
    all_passed = all(
        r.get('passed', True) if isinstance(r, dict) else True
        for r in results.values() if r is not None
    )

    # 检查嵌套字典
    if results['model_size']:
        for v in results['model_size'].values():
            if isinstance(v, dict) and not v.get('passed', True):
                all_passed = False

    print(f"\n{'='*60}")
    print(f"  总结: {'✅ 全部通过' if all_passed else '❌ 存在未通过项'}")
    print(f"{'='*60}")

    # 保存结果
    output_file = DL_MODELS_DIR / "benchmark_results.json"

    def convert(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\n📄 结果保存: {output_file}")

    return results


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
