import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import io
import base64
import io
import base64

# ==========================================
# 1. CORE UTILITIES
# ==========================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients, self.activations = None, None
        
        # Hooks for capturing activations and gradients
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Weight gradients by spatial average
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach().clone()
        
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        
        probs = torch.softmax(output, dim=1)
        return heatmap, class_idx, probs[0].tolist()

# ==========================================
# 2. IMAGE PROCESSING
# ==========================================

def get_transforms():
    """Matches the validation transforms from neural-nexus-final.ipynb"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def apply_clahe(img_np):
    """Optional clinical enhancement filter"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply to each channel
    enhanced = np.stack([clahe.apply(img_np[:,:,i]) for i in range(3)], axis=-1)
    return enhanced

def overlay_heatmap(img_np, heatmap, alpha=0.5):
    """Combines MRI scan with AI attention map"""
    heatmap_res = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    combined = cv2.addWeighted(img_np, 1 - alpha, heatmap_rgb, alpha, 0)
    return combined

# ==========================================
# 3. ENCODING UTILS
# ==========================================

def numpy_to_base64(img_np):
    """Converts numpy image (RGB) to Base64 string for UI display"""
    pil_img = Image.fromarray(img_np.astype('uint8'))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# ==========================================
# 4. CLINICAL RISK EXTRACTION
# ==========================================

def extract_risk_metrics(heatmap, confidence, label):
    """
    Computes mathematical clinical risk metrics from the AI attention map.
    """
    if label == "No Tumor":
        return {
            "entropy": 0.0,
            "irregularity_ratio": 0.0,
            "activation_area": 0.0,
            "risk_score": 5
        }
        
    flat_heat = heatmap.flatten()
    active_pixels = flat_heat[flat_heat > 0.15] # Threshold focus area
    
    # 1. Image Entropy (Measures morphological irregularity)
    if len(active_pixels) > 0:
        p = active_pixels / np.sum(active_pixels)
        entropy = -np.sum(p * np.log2(p + 1e-9))
    else:
        entropy = 0.0
        
    # Normalize entropy roughly to 0-1
    normalized_entropy = min(max((entropy - 4.0) / 8.0, 0.0), 1.0)
        
    # 2. Activation Size
    activation_area = len(active_pixels) / len(flat_heat)
    
    # 3. Composite Risk Score (1-100)
    risk = 0.0
    if label == "Glioma": risk += 45
    elif label == "Meningioma": risk += 25
    elif label == "Pituitary": risk += 20
        
    risk += confidence * 15
    risk += min(activation_area * 150, 20) # Caps at 20 points for size
    risk += normalized_entropy * 20
    
    final_risk = min(max(int(risk), 10), 99)
    
    return {
        "entropy": round(float(entropy), 2),
        "irregularity_ratio": round(float(normalized_entropy), 2),
        "activation_area": round(float(activation_area), 3),
        "risk_score": final_risk
    }

