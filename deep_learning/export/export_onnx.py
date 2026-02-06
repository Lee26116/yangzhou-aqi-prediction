# -*- coding: utf-8 -*-
"""
ONNX 模型导出
导出 AQIPredictorInference（无 attention 输出）
用 onnxruntime 验证正确性
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from deep_learning.dl_config import (
    DL_MODELS_DIR, DL_EXPORTED_DIR, DL_SEQUENCES_DIR,
    ONNX_CONFIG, MODEL_CONFIG, SEQUENCE_LENGTH
)
from deep_learning.models.lstm_attention import (
    build_model, AQIPredictorInference
)


def export_to_onnx():
    """导出模型为 ONNX 格式"""
    print("=" * 60)
    print("  ONNX 模型导出")
    print("=" * 60)

    # 加载最佳模型
    print("\n📦 加载最佳模型...")
    checkpoint_path = DL_MODELS_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"模型不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    input_size = checkpoint['input_size']

    model = build_model(input_size, checkpoint.get('model_config', MODEL_CONFIG))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建推理模型（不返回 attention）
    print("\n🔧 创建推理模型...")
    inference_model = AQIPredictorInference(model)
    inference_model.eval()

    # 创建示例输入
    dummy_input = torch.randn(1, SEQUENCE_LENGTH, input_size)

    # 导出 ONNX
    onnx_path = DL_EXPORTED_DIR / ONNX_CONFIG['model_filename']
    print(f"\n📤 导出 ONNX 到: {onnx_path}")

    torch.onnx.export(
        inference_model,
        dummy_input,
        str(onnx_path),
        opset_version=ONNX_CONFIG['opset_version'],
        input_names=['input'],
        output_names=['predictions'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'predictions': {0: 'batch_size'},
        },
        dynamo=False,
    )

    print(f"   文件大小: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")

    # 验证 ONNX 模型
    print("\n🔍 验证 ONNX 模型...")
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("   ✅ ONNX 模型结构验证通过")
    except ImportError:
        print("   ⚠️ onnx 未安装，跳过结构验证")

    # 用 onnxruntime 验证正确性
    print("\n🔍 OnnxRuntime 推理验证...")
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: dummy_input.numpy()}
        ort_outputs = session.run(None, ort_inputs)

        # 对比 PyTorch 推理
        with torch.no_grad():
            torch_output = inference_model(dummy_input).numpy()

        max_diff = np.max(np.abs(torch_output - ort_outputs[0]))
        print(f"   PyTorch vs ONNX 最大差异: {max_diff:.6f}")

        if max_diff < 1e-4:
            print("   ✅ 推理结果一致性验证通过")
        else:
            print(f"   ⚠️ 差异较大，请检查 (max_diff={max_diff:.6f})")

    except ImportError:
        print("   ⚠️ onnxruntime 未安装，跳过推理验证")

    # 保存导出元数据
    import json
    meta = {
        'input_size': input_size,
        'sequence_length': SEQUENCE_LENGTH,
        'opset_version': ONNX_CONFIG['opset_version'],
        'model_file': ONNX_CONFIG['model_filename'],
        'source_epoch': checkpoint['epoch'],
        'source_val_loss': float(checkpoint['val_loss']),
    }
    meta_file = DL_EXPORTED_DIR / "export_metadata.json"
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ ONNX 导出完成!")
    return str(onnx_path)


def main():
    export_to_onnx()


if __name__ == "__main__":
    main()
