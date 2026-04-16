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
    numpy_to_base64
)
from report import generate_clinical_report
from uncertainty import mc_dropout_inference, generate_uncertainty_heatmap

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

# Global model and CAM instance
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
    images: Dict[str, str]  # Base64 strings: original, enhanced, heatmap
    tumor_location: Dict[str, float] = None

class UncertaintyResult(BaseModel):
    """Extended analysis result with MC Dropout uncertainty metrics."""
    label: str
    confidence: float
    probabilities: Dict[str, float]
    images: Dict[str, str]
    tumor_location: Optional[Dict[str, float]] = None
    # --- Uncertainty Metrics ---
    mean_confidence: float
    std_confidence: float
    mean_probs: Dict[str, float]
    std_probs: Dict[str, float]
    reliability_score: float
    entropy: float
    mutual_information: float
    prediction_counts: Dict[str, int]
    n_samples: int

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
        
        # 3. Inference & Grad-CAM
        heatmap, idx, all_probs = cam_engine.generate(input_tensor)
        
        # 4. Image Generation
        enhanced_np = apply_clahe(img_np)
        heatmap_overlay = overlay_heatmap(img_np, heatmap, alpha=0.5)
        
        # 5. Extract tumor center (highest activation)
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        tumor_loc = {
            "x": float(x / heatmap.shape[1]),
            "y": float(y / heatmap.shape[0])
        } if CLASSES[idx] != "No Tumor" else None

        # 6. Result Packaging
        results = {
            "label": CLASSES[idx],
            "confidence": all_probs[idx],
            "probabilities": {CLASSES[i]: all_probs[i] for i in range(len(CLASSES))},
            "images": {
                "original": numpy_to_base64(img_np),
                "enhanced": numpy_to_base64(enhanced_np),
                "heatmap": numpy_to_base64(heatmap_overlay)
            },
            "tumor_location": tumor_loc
        }
        
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/analyze-uncertainty", response_model=UncertaintyResult)
async def analyze_mri_uncertainty(file: UploadFile = File(...)):
    """
    Advanced analysis with Monte Carlo Dropout uncertainty quantification.
    Runs N stochastic forward passes to estimate prediction reliability.
    """
    if not model_engine:
        raise HTTPException(status_code=503, detail="AI Core is currently offline.")

    try:
        # 1. Load and Preprocess (same as standard analysis)
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)

        transform = get_transforms()
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # 2. Standard Inference & Grad-CAM (for base results)
        heatmap, idx, all_probs = cam_engine.generate(input_tensor)

        # 3. Image Generation
        enhanced_np = apply_clahe(img_np)
        heatmap_overlay = overlay_heatmap(img_np, heatmap, alpha=0.5)

        # 4. Tumor Location
        y_loc, x_loc = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        tumor_loc = {
            "x": float(x_loc / heatmap.shape[1]),
            "y": float(y_loc / heatmap.shape[0])
        } if CLASSES[idx] != "No Tumor" else None

        # 5. MC Dropout Uncertainty Analysis
        mc_results = mc_dropout_inference(
            model_engine,
            input_tensor,
            n_samples=30,
            classes=CLASSES
        )

        # 6. Uncertainty Heatmap
        unc_heatmap = generate_uncertainty_heatmap(
            model_engine,
            input_tensor,
            model_engine.base_model.layer4,
            n_samples=15
        )
        unc_heatmap_overlay = overlay_heatmap(img_np, unc_heatmap, alpha=0.6)

        # 7. Package Results
        results = {
            "label": CLASSES[idx],
            "confidence": all_probs[idx],
            "probabilities": {CLASSES[i]: all_probs[i] for i in range(len(CLASSES))},
            "images": {
                "original": numpy_to_base64(img_np),
                "enhanced": numpy_to_base64(enhanced_np),
                "heatmap": numpy_to_base64(heatmap_overlay),
                "uncertainty": numpy_to_base64(unc_heatmap_overlay),
            },
            "tumor_location": tumor_loc,
            # Uncertainty fields
            "mean_confidence": mc_results["mean_confidence"],
            "std_confidence": mc_results["std_confidence"],
            "mean_probs": mc_results["mean_probs"],
            "std_probs": mc_results["std_probs"],
            "reliability_score": mc_results["reliability_score"],
            "entropy": mc_results["entropy"],
            "mutual_information": mc_results["mutual_information"],
            "prediction_counts": mc_results["prediction_counts"],
            "n_samples": mc_results["n_samples"],
        }

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Uncertainty analysis failed: {str(e)}")

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
