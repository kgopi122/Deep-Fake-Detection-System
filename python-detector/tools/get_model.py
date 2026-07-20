import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os

def export_model(model_id, output_filename):
    print(f"\n🚀 Processing: {model_id} -> {output_filename}")
    
    try:
        print(f"   📥 Downloading config & weights...")
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
        model.eval()
        
        print(f"   🔄 Converting to ONNX...")
        # Create dummy input based on model config (usually 224x224)
        h, w = 224, 224
        if hasattr(processor, "size"):
            if "height" in processor.size:
                h = processor.size["height"]
                w = processor.size["width"]
        
        dummy_input = torch.randn(1, 3, h, w)
        
        # Robust path finding:
        # Script is in: .../python-detector/tools/get_model.py
        # We want:      .../veritrue-backend/models/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up 2 levels (tools -> python-detector -> veritrue root) then into veritrue-backend
        base_dir = os.path.abspath(os.path.join(script_dir, "../../"))
        output_path = os.path.join(base_dir, "veritrue-backend", "models", output_filename)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        torch.onnx.export(
            model, 
            dummy_input, 
            output_path, 
            export_params=True, 
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"   ✅ Saved to: {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")

def main():
    # 1. Face Specialist (DeepFake Detector v2 - Good at faces)
    export_model("prithivMLmods/Deep-Fake-Detector-v2-Model", "face.onnx")
    
    # 2. Artifact Specialist (AI vs Human - Good at Midjourney/GenAI artifacts)
    export_model("Ateeqq/ai-vs-human-image-detector", "artifacts.onnx")
    
    # 3. Noise/Texture Specialist (Distilled Model - Good secondary voter)
    export_model("jacoballessio/ai-image-detect-distilled", "noise.onnx")

    print("\n✨ All models processed! Please restart the backend.")

if __name__ == "__main__":
    main()
