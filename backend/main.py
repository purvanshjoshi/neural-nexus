from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
import numpy as np
import cv2
from PIL import Image
import io
import os

from model import load_official_model
from utils import (
    GradCAM,
    get_transforms, 
    apply_clahe, 
    overlay_heatmap, 
    numpy_to_base64,
    extract_risk_metrics,
    get_score_cam
)
from report import generate_clinical_report
from llm_engine import generate_clinical_narrative

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

app = FastAPI(title="Neural Nexus | AI Diagnostic Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "best_tumor_model.pth")
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

model_engine = None
cam_engine = None

@app.on_event("startup")
async def startup_event():
    global model_engine, cam_engine
    if os.path.exists(WEIGHTS_PATH):
        try:
            model_engine = load_official_model(WEIGHTS_PATH, DEVICE)
            cam_engine = GradCAM(model_engine, model_engine.base_model.layer4)
            print("Neural Nexus AI Core Loaded and Online.")
        except Exception as e:
            print(f"Failed to load AI Core: {str(e)}")
    else:
        print(f"Warning: Model weights not found at {WEIGHTS_PATH}")

# ==========================================
# 2. MODELS & SCHEMAS
# ==========================================

class AnalysisResult(BaseModel):
    label: str
    confidence: float
    probabilities: Dict[str, float]
    images: Dict[str, str]
    tumor_location: Optional[Dict[str, float]] = None
    risk_metrics: Optional[Dict] = None
    clinical_narrative: Optional[str] = None
    uncertainty: float = 0.0

# ==========================================
# 3. ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    return {
        "message": "Neural Nexus AI Diagnostic Core is Online",
        "version": "1.1.0",
        "features": ["BioMistral Narrative", "Risk Metrics", "Score-CAM", "MC Dropout Uncertainty"]
    }

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_mri(file: UploadFile = File(...)):
    if not model_engine:
        raise HTTPException(status_code=503, detail="AI Core is currently offline.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        
        transform = get_transforms()
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # 1. Inference & XAI (Grad-CAM + MC Dropout)
        heatmap, idx, all_probs, uncertainty = cam_engine.generate(input_tensor, mc_samples=10)
        
        # 2. Extract Loc & Risk
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        tumor_loc = {
            "x": float(x / heatmap.shape[1]),
            "y": float(y / heatmap.shape[0])
        } if CLASSES[idx] != "No Tumor" else None
        
        risk_data = extract_risk_metrics(heatmap, float(all_probs[idx]), CLASSES[idx])
        
        # 3. Image Generation
        enhanced_np = apply_clahe(img_np)
        heatmap_overlay = overlay_heatmap(img_np, heatmap)
        
        results = {
            "label": CLASSES[idx],
            "confidence": float(all_probs[idx]),
            "probabilities": {CLASSES[i]: float(all_probs[i]) for i in range(len(CLASSES))},
            "images": {
                "original": numpy_to_base64(img_np),
                "enhanced": numpy_to_base64(enhanced_np),
                "heatmap": numpy_to_base64(heatmap_overlay)
            },
            "tumor_location": tumor_loc,
            "risk_metrics": risk_data,
            "uncertainty": uncertainty
        }
        
        # 4. BioMistral Narrative Logic
        try:
            results["clinical_narrative"] = generate_clinical_narrative(results, risk_data)
        except Exception as e:
            print(f"Narrative generation failed: {e}")
            results["clinical_narrative"] = f"BioMistral API Delay. Note: Primary finding is {results['label']}."
            
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {str(e)}")

@app.post("/api/report")
async def get_report(data: AnalysisResult):
    try:
        pdf_bytes = generate_clinical_report(data.model_dump())
        return Response(
            content=bytes(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=Neural_Nexus_Report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report failed: {str(e)}")

@app.post("/api/narrative")
async def get_narrative(data: Dict):
    """Stand-alone narrative generator for chat/oracle."""
    try:
        risk_metrics = data.get("risk_metrics", {})
        narrative = generate_clinical_narrative(data, risk_metrics)
        return {"narrative": narrative}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Narrative error: {str(e)}")

@app.post("/api/scorecam")
async def get_scorecam(file: UploadFile = File(...)):
    if not model_engine:
        raise HTTPException(status_code=503, detail="AI Core offline.")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        input_tensor = get_transforms()(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            class_idx = model_engine(input_tensor).argmax(dim=1).item()
        
        score_map = get_score_cam(model_engine, model_engine.base_model.layer4, input_tensor, class_idx)
        return {"heatmap": numpy_to_base64(overlay_heatmap(img_np, score_map))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Score-CAM error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
