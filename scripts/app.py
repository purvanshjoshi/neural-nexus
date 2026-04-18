import gradio as gr
import torch
import numpy as np
from PIL import Image
from inference import load_model, preprocess_image, GradCAM, overlay_heatmap

# ==========================================
# CONFIGURATION (Change for Kaggle)
# ==========================================
MODEL_PATH = "best_model.pth"  # Local path
# For Kaggle: Use "/kaggle/working/best_model.pth" or similar
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# INITIALIZATION
# ==========================================
print(f"Loading system core... [Device: {DEVICE.upper()}]")
model = load_model(MODEL_PATH, device=DEVICE)
target_layer = model.backbone.features[-1]  # Last conv layer
cam_engine = GradCAM(model, target_layer)

def analyze_scan(img, use_clahe, heatmap_alpha):
    if img is None:
        return None, {}, "Please upload an MRI scan."
    
    # 1. Preprocess
    tensor, processed_img = preprocess_image(img, use_clahe=use_clahe)
    tensor = tensor.to(DEVICE)
    
    # 2. Inference & Grad-CAM
    heatmap, class_idx, conf, all_probs_raw = cam_engine.generate(tensor)
    
    # 3. Format Label Output
    confidence_map = {CLASSES[i]: float(all_probs_raw[f"Class {i}"]) for i in range(len(CLASSES))}
    
    # 4. Generate Overlays
    gradcam_overlay = overlay_heatmap(processed_img, heatmap, alpha=heatmap_alpha)
    
    # 5. Compile Gallery
    gallery_images = [
        (img, "Original Scan"),
        (processed_img, "CLAHE Enhanced" if use_clahe else "Raw Input"),
        (gradcam_overlay, f"Grad-CAM (Focus on {CLASSES[class_idx]})")
    ]
    
    status = f"Analysis Complete. Detection: {CLASSES[class_idx]} ({conf:.2%})"
    return gallery_images, confidence_map, status

# ==========================================
# ASTRAGUARD THEME & UI
# ==========================================
theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_width="1px",
    block_title_text_color="*primary_400",
)

with gr.Blocks(theme=theme, title="AstraGuard | Brain Tumor Analysis") as demo:
    gr.Markdown(
        """
        # 🧠 AstraGuard AI: Brain Tumor Mission Control
        **SOTA EfficientNet-V2 Diagnostics for Clinical Support**
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload MRI Scan", type="pil")
            
            with gr.Group():
                gr.Markdown("### ⚙️ Engine Diagnostics")
                clahe_toggle = gr.Checkbox(label="Enable CLAHE Preprocessing", value=True)
                alpha_slider = gr.Slider(label="Grad-CAM Transparency (Alpha)", minimum=0.1, maximum=0.9, value=0.5)
            
            analyze_btn = gr.Button("🚀 EXECUTE ANALYSIS", variant="primary")
            
        with gr.Column(scale=2):
            output_status = gr.Textbox(label="System Status", interactive=False)
            output_label = gr.Label(label="Classification Probability", num_top_classes=4)
            output_gallery = gr.Gallery(label="Visual Analysis Results", columns=3, height="auto")
    
    analyze_btn.click(
        fn=analyze_scan,
        inputs=[input_img, clahe_toggle, alpha_slider],
        outputs=[output_gallery, output_label, output_status]
    )
    
    gr.Markdown(
        """
        ---
        **Disclaimer**: This AI system is for research purposes and clinical decision support only. Final diagnosis must be performed by a certified radiologist.
        *Powered by AstraGuard AI Systems | Model: EfficientNet-V2-S*
        """
    )

if __name__ == "__main__":
    demo.launch(share=True)
