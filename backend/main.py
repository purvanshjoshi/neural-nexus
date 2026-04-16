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
    get_transforms, 
    apply_clahe, 
    overlay_heatmap, 
    numpy_to_base64,
    extract_risk_metrics,
    generate_multi_xai
)
from report import generate_clinical_report
from llm_engine import generate_clinical_narrative

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

app = FastAPI(title="Neural Nexus | AI Diagnostic Core")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "best_tumor_model.pth")
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Global model and XAI target layers
model_engine = None
target_layers = None

@app.on_event("startup")
async def startup_event():
    global model_engine, cam_engine
    if os.path.exists(WEIGHTS_PATH):
        try:
            model_engine = load_official_model(WEIGHTS_PATH, DEVICE)
            # ResNet-50 target layer is layer4 (the last convolutional block)
            target_layers = [model_engine.base_model.layer4]
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
    images: Dict[str, str]  # Base64 strings: original, enhanced, heatmap
    tumor_location: Optional[Dict[str, float]] = None
    risk_metrics: Optional[Dict] = None
    clinical_narrative: Optional[str] = None

# ==========================================
# 3. ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    return {
        "message": "Neural Nexus AI Diagnostic Core is Online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "analyze": "/api/analyze (POST)",
            "report": "/api/report (POST)"
        },
        "documentation": "/docs"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "online" if model_engine else "offline",
        "device": str(DEVICE),
        "engine": "Neural Nexus ResNet-50 v1.0"
    }

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_mri(file: UploadFile = File(...)):
    if not model_engine:
        raise HTTPException(status_code=503, detail="AI Core is currently offline.")

    try:
        # 1. Load and Preprocess
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        
        # 2. Transformations
        transform = get_transforms()
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # 3. Inference & Multi-XAI Analysis
        with torch.no_grad():
            output = model_engine(input_tensor)
            probs = torch.softmax(output, dim=1)
            all_probs = probs[0].tolist()
            idx = torch.argmax(output, dim=1).item()
            
        xai_maps = generate_multi_xai(model_engine, input_tensor, target_layers, idx)
        
        # 4. Image Processing & Overlays
        enhanced_np = apply_clahe(img_np)
        
        # We generate overlays for all XAI methods
        overlays = {
            "gradcam": numpy_to_base64(overlay_heatmap(img_np, xai_maps['gradcam'])),
            "gradcam_plus": numpy_to_base64(overlay_heatmap(img_np, xai_maps['gradcam_plus'])),
            "layercam": numpy_to_base64(overlay_heatmap(img_np, xai_maps['layercam'])),
            "consensus": numpy_to_base64(overlay_heatmap(img_np, xai_maps['consensus']))
        }
        
        # 5. Extract tumor center (using consensus for stability)
        heatmap = xai_maps['consensus']
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        tumor_loc = {
            "x": float(x / heatmap.shape[1]),
            "y": float(y / heatmap.shape[0])
        } if CLASSES[idx] != "No Tumor" else None

        # 6. Risk metrics computation (uses consensus heatmap)
        risk_metrics = extract_risk_metrics(heatmap, float(all_probs[idx]), CLASSES[idx])

        # 7. Result Packaging
        results = {
            "label": CLASSES[idx],
            "confidence": float(all_probs[idx]),
            "probabilities": {CLASSES[i]: float(all_probs[i]) for i in range(len(CLASSES))},
            "images": {
                "original": numpy_to_base64(img_np),
                "enhanced": numpy_to_base64(enhanced_np),
                "heatmap": overlays["gradcam"], # Default for backward compatibility
                **overlays
            },
            "tumor_location": tumor_loc,
            "risk_metrics": risk_metrics
        }
        
        # 8. Generate Clinical Risk Explanation
        try:
            results["clinical_narrative"] = generate_clinical_narrative(results, risk_metrics)
        except Exception as e:
            print(f"Risk Explainer generation failed: {e}")
            results["clinical_narrative"] = f"BioMistral API Error. Note: Primary diagnosis is {results['label']}."
        
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/report")
async def get_report(data: AnalysisResult):
    try:
        # Pydantic V2 use model_dump()
        pdf_bytes = generate_clinical_report(data.model_dump())
        return Response(
            content=bytes(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=Neural_Nexus_Report.pdf"}
        )
    except Exception as e:
        print(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

# ==========================================
# 4. EXECUTION
# ==========================================

if __name__ == "__main__":
    import uvicorn
    # Use PORT environment variable for cloud compatibility (e.g. Hugging Face uses 7860)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
