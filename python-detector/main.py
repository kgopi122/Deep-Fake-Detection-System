from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import warnings

# Suppress PIL warnings for image conversions
warnings.filterwarnings("ignore", category=UserWarning)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://localhost:5173,http://localhost:5174,http://localhost:8000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Global Setup (Dual Engines)
print("Loading Photo Model (Organika/sdxl-detector)...")
photo_processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
photo_model = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")
photo_model.eval()

print("Loading Art Model (umm-maybe/AI-image-detector)...")
art_processor = AutoImageProcessor.from_pretrained("umm-maybe/AI-image-detector")
art_model = AutoModelForImageClassification.from_pretrained("umm-maybe/AI-image-detector")
art_model.eval()


def extract_fake_score(outputs, model) -> float:
    """Extracts the fake probability using exact dynamic Softmax loop"""
    # 1. Get raw probabilities via Softmax
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    # 2. Dynamically find the 'fake' class index
    fake_idx = 1 # Default to 1 if we can't find it
    for idx, label in model.config.id2label.items():
        label_lower = label.lower()
        if any(keyword in label_lower for keyword in ['fake', 'ai', 'synthetic', 'generated', 'artificial']):
            fake_idx = idx
            break

    # 3. Extract the probability and convert strictly to a 0-100 percentage
    return float(probs[fake_idx].item() * 100.0)


@app.post("/api/analyze/pixel")
async def analyze_pixel(file: UploadFile = File(...)):
    contents = await file.read()
    
    # 2. The Robust Image Loading Pipeline (Transparency Fix)
    img = Image.open(io.BytesIO(contents))
    
    if img.mode == 'P' or img.mode in ('RGBA', 'LA', 'PA'):
        img = img.convert('RGBA')
    
    img = img.convert('RGB')
    
    # 2. The Inference Pipeline (Concurrent Model Execution)
    photo_inputs = photo_processor(images=img, return_tensors="pt")
    art_inputs = art_processor(images=img, return_tensors="pt")
    
    with torch.no_grad():
        photo_outputs = photo_model(**photo_inputs)
        art_outputs = art_model(**art_inputs)

    # 3. Softmax Extraction (The Math)
    photo_score = extract_fake_score(photo_outputs, photo_model)
    art_score = extract_fake_score(art_outputs, art_model)

    # 4. Max-Confidence Logic
    composite_score = max(photo_score, art_score)
    
    if photo_score > art_score:
        file_type_routed = "PHOTO_ENGINE_MATCH"
        detected_label_string = photo_model.config.id2label[photo_outputs.logits.argmax(-1).item()]
    else:
        file_type_routed = "ART_ENGINE_MATCH"
        detected_label_string = art_model.config.id2label[art_outputs.logits.argmax(-1).item()]

    is_fake = True if composite_score >= 85.0 else False
    
    # 5. Cleanup & Contract
    print(f"PHOTO SCORE: {photo_score} | ART SCORE: {art_score}")
    
    return {
        "is_fake": is_fake,
        "composite_score": round(composite_score, 2),
        "vit_fake_score": round(composite_score, 2),
        "srm_physics_score": 0.0,
        "detected_label_string": str(detected_label_string),
        "file_type_routed": file_type_routed
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)