import os
import sys
import argparse

# Add workspace root to sys.path to enable unpickling serialized modules
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Ensure UTF-8 stdout/stderr on Windows to handle unicode output from torch.onnx
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


class BaselineDeepfakeDetector(nn.Module):
    """
    Baseline PyTorch CNN model architecture for deepfake / AIGC image detection.
    Inputs: [batch_size, 3, 224, 224]
    Outputs: Logits for [real (idx 0), fake (idx 1)]
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def find_checkpoint(explicit_path: str = None) -> str:
    """Finds a valid PyTorch model checkpoint or raises FileNotFoundError."""
    if explicit_path:
        path = os.path.abspath(explicit_path)
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(f"Specified PyTorch checkpoint file '{explicit_path}' does not exist.")

    search_paths = [
        os.path.join(root_dir, "models", "model.pt"),
        os.path.join(root_dir, "models", "model.pth"),
        os.path.join(root_dir, "weights.pt"),
        os.path.join(root_dir, "weights.pth"),
        os.path.join(root_dir, "scripts", "model.pt"),
        os.path.join(root_dir, "scripts", "model.pth"),
        os.path.join(root_dir, "python-detector", "model.pt"),
        os.path.join(root_dir, "python-detector", "model.pth"),
        os.path.join(root_dir, "python-detector", "veritrue_detector", "model.pt"),
        os.path.join(root_dir, "veritrue-backend", "src", "main", "resources", "models", "veritrue.pt"),
        os.path.join(root_dir, "veritrue-backend", "src", "main", "resources", "models", "veritrue.pth"),
    ]

    for path in search_paths:
        if os.path.isfile(path):
            print(f"Auto-discovered PyTorch model checkpoint at: {path}")
            return path

    raise FileNotFoundError(
        "No PyTorch model checkpoint provided via --pt-path and none found in common project locations "
        "(models/, scripts/, python-detector/).\n"
        "Please provide a trained .pt or .pth checkpoint file using: python scripts/export_quantize.py --pt-path <PATH_TO_MODEL.pt>"
    )


def load_pytorch_model(pt_path: str) -> nn.Module:
    """Flexible loader for TorchScript, serialized nn.Module, or state_dict checkpoints."""
    print(f"Loading PyTorch checkpoint from: {pt_path}")

    # 1. Attempt loading as TorchScript model
    try:
        ts_model = torch.jit.load(pt_path, map_location="cpu")
        ts_model.eval()
        print("Successfully loaded model as TorchScript module.")
        return ts_model
    except Exception:
        pass

    # 2. Attempt standard torch.load (handling weights_only in PyTorch 2.6+)
    try:
        try:
            loaded = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception:
            loaded = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to parse PyTorch file '{pt_path}': {e}")

    # Case A: Loaded object is directly an nn.Module instance
    if isinstance(loaded, nn.Module):
        loaded.eval()
        print("Successfully loaded full PyTorch nn.Module object.")
        return loaded

    # Case B: Loaded object is a state_dict or dictionary containing state_dict
    if isinstance(loaded, dict):
        sd = loaded
        for key in ["state_dict", "model_state_dict", "model", "net", "params"]:
            if key in loaded and isinstance(loaded[key], dict):
                sd = loaded[key]
                break

        # Remove "module." prefix if saved from DataParallel / DistributedDataParallel
        cleaned_sd = {}
        for k, v in sd.items():
            new_key = k[7:] if k.startswith("module.") else k
            cleaned_sd[new_key] = v

        # Instantiate target model architecture
        model = BaselineDeepfakeDetector()
        try:
            model.load_state_dict(cleaned_sd, strict=True)
            print("Successfully loaded state_dict into BaselineDeepfakeDetector with strict parameter matching.")
        except Exception as e_strict:
            print(f"Strict state_dict loading failed ({e_strict}). Attempting non-strict loading...")
            try:
                model.load_state_dict(cleaned_sd, strict=False)
                print("Loaded state_dict with strict=False.")
            except Exception as e_nonstrict:
                raise ValueError(f"Failed to load state_dict into BaselineDeepfakeDetector: {e_nonstrict}")

        model.eval()
        return model

    raise ValueError(f"Unrecognized object type loaded from '{pt_path}': {type(loaded)}")


def verify_onnx_model(onnx_path: str):
    """Performs post-export sanity verification using ONNX Runtime."""
    print("\n" + "=" * 50)
    print("RUNNING POST-EXPORT ONNX SANITY VERIFICATION...")
    print("=" * 50)

    try:
        session = ort.InferenceSession(onnx_path)
        input_info = session.get_inputs()[0]
        input_name = input_info.name

        # Prepare dummy image tensor (1, 3, 224, 224)
        dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})

        logits = outputs[0]
        # Calculate Softmax probabilities over class logits
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        pred_idx = int(np.argmax(probs[0]))
        real_pct = probs[0][0] * 100.0 if len(probs[0]) > 0 else 0.0
        fake_pct = probs[0][1] * 100.0 if len(probs[0]) > 1 else 0.0

        print(f"Quantized Model Path: {onnx_path}")
        print(f"Input Name: '{input_name}', Shape: {input_info.shape}")
        print(f"Raw Output Logits: {logits.tolist()}")
        print(f"Softmax Probabilities: Real (Class 0): {real_pct:.2f}%, Fake (Class 1): {fake_pct:.2f}%")
        print(f"Predicted Class Index: {pred_idx} ({'Fake' if pred_idx == 1 else 'Real'})")
        print("=" * 50)
        print("ONNX MODEL SANITY VERIFICATION PASSED SUCCESSFULLY ✅")
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"ERROR: ONNX Sanity Verification failed for '{onnx_path}': {e}")
        raise


def export_and_quantize(pt_path: str = None, output_dir: str = None):
    # 1. Fail-fast checkpoint search & validation
    checkpoint_path = find_checkpoint(pt_path)

    # 2. Load PyTorch model
    model = load_pytorch_model(checkpoint_path)

    # 3. Determine output directory inside Spring Boot resources
    if output_dir is None:
        target_dir = os.path.join(root_dir, "veritrue-backend", "src", "main", "resources", "models")
    else:
        target_dir = output_dir

    os.makedirs(target_dir, exist_ok=True)
    output_onnx_path = os.path.join(target_dir, "veritrue_int8.onnx")
    temp_fp32_onnx = os.path.join(target_dir, "temp_veritrue_fp32.onnx")

    print(f"Target INT8 ONNX Model Path: {output_onnx_path}")

    # 4. Export FP32 ONNX model
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    print(f"Exporting PyTorch model to FP32 ONNX: {temp_fp32_onnx}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            temp_fp32_onnx,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            temp_fp32_onnx,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )

    # 5. Dynamic Quantization to INT8 (QUInt8)
    print(f"Quantizing ONNX model to QUInt8 INT8: {output_onnx_path}...")
    quantize_dynamic(
        model_input=temp_fp32_onnx,
        model_output=output_onnx_path,
        weight_type=QuantType.QUInt8
    )

    if os.path.exists(temp_fp32_onnx):
        os.remove(temp_fp32_onnx)

    print(f"Successfully generated INT8 ONNX model at: {output_onnx_path}")

    # 6. Post-Export ONNX Sanity Verification
    verify_onnx_model(output_onnx_path)

    return output_onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch .pt model to quantized INT8 ONNX model for VeriTrue.")
    parser.add_argument("--pt-path", type=str, default=None, help="Path to input PyTorch model (.pt or .pth)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save veritrue_int8.onnx")

    args = parser.parse_args()
    export_and_quantize(pt_path=args.pt_path, output_dir=args.output_dir)
