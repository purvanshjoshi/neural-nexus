import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import io
import base64
from scipy.stats import entropy

# ==========================================
# 1. CORE UTILITIES (XAI ENGINES)
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

    def generate(self, input_tensor, class_idx=None, mc_samples=0):
        """
        Generates Grad-CAM heatmap with optional MC Dropout for uncertainty.
        """
        self.model.eval()
        
        # Standard Pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Grad-CAM Heatmap
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach().clone()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        
        # Uncertainty Quantization (MC Dropout)
        uncertainty = 0.0
        if mc_samples > 0:
            # Enable dropout specifically
            def enable_dropout(m):
                if type(m) == torch.nn.Dropout:
                    m.train()
            
            self.model.apply(enable_dropout)
            samples = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    samp_out = torch.softmax(self.model(input_tensor), dim=1)
                    samples.append(samp_out[0, class_idx].item())
            
            uncertainty = float(np.std(samples))
            self.model.eval() # Reset to eval
            
        probs = torch.softmax(output, dim=1)
        return heatmap, class_idx, probs[0].tolist(), uncertainty

def get_score_cam(model, target_layer, input_tensor, class_idx):
    """
    Score-CAM: Gradient-free visualization using model confidence weighting.
    """
    model.eval()
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output)
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        output = model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        # Get activations
        act = activations[0] # [1, C, H, W]
        handle.remove()
        
        # Upsample activations to input size
        upsampled = torch.nn.functional.interpolate(act, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # Normalize each channel
        max_val = upsampled.view(upsampled.size(0), upsampled.size(1), -1).max(dim=2)[0].view(upsampled.size(0), upsampled.size(1), 1, 1)
        min_val = upsampled.view(upsampled.size(0), upsampled.size(1), -1).min(dim=2)[0].view(upsampled.size(0), upsampled.size(1), 1, 1)
        upsampled = (upsampled - min_val) / (max_val - min_val + 1e-8)
        
        # Get scores for masked inputs
        weights = []
        for i in range(act.shape[1]):
            m_in = input_tensor * upsampled[0:1, i:i+1, :, :]
            score = torch.softmax(model(m_in), dim=1)
            weights.append(score[0, class_idx].item())
            
        weights = torch.tensor(weights).to(input_tensor.device)
        score_map = torch.sum(weights.view(1, -1, 1, 1) * act, dim=1).squeeze()
        
        score_map = np.maximum(score_map.cpu().numpy(), 0)
        score_map /= (np.max(score_map) + 1e-8)
        
        return score_map

# ==========================================
# 2. IMAGE PROCESSING UTILS
# ==========================================

def get_transforms():
    """Official Neural Nexus preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def apply_clahe(img_np):
    """Clinical enhancement for radiological contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply to each channel independently
    enhanced = np.stack([clahe.apply(img_np[:,:,i]) for i in range(3)], axis=-1)
    return enhanced

def overlay_heatmap(img_np, heatmap, alpha=0.5):
    """Overlay tool for XAI attention visualization."""
    heatmap_res = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_res), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    combined = cv2.addWeighted(img_np, 1 - alpha, heatmap_rgb, alpha, 0)
    return combined

# ==========================================
# 3. CLINICAL RISK ANALYTICS
# ==========================================

def extract_risk_metrics(heatmap, confidence, label):
    """
    Unified Risk Metric Suite. Combines morphological features (entropy, asymmetry)
    with diagnostic weighting (Glioma risk, etc.)
    """
    if label == "No Tumor":
        return {
            "entropy": 0.0,
            "irregularity_ratio": 0.0,
            "activation_area": 0.0,
            "asymmetry": 0.0,
            "risk_score": 5
        }
        
    width = heatmap.shape[1]
    flat_heat = heatmap.flatten()
    active_mask = (heatmap > 0.15).astype(np.float32)
    active_pixels = flat_heat[flat_heat > 0.15]
    
    # 1. Morphological Entropy (Chaos)
    if len(active_pixels) > 0:
        p = active_pixels / np.sum(active_pixels)
        ent_val = -np.sum(p * np.log2(p + 1e-9))
    else:
        ent_val = 0.0
    
    normalized_entropy = min(max((ent_val - 4.0) / 8.0, 0.0), 1.0)
        
    # 2. Activation Size (Relative Area)
    activation_area = len(active_pixels) / len(flat_heat)
    
    # 3. Lateral Asymmetry
    left_side = heatmap[:, :width//2]
    right_side = heatmap[:, width//2:]
    l_mass, r_mass = np.sum(left_side), np.sum(right_side)
    asymmetry = float(abs(l_mass - r_mass) / (l_mass + r_mass + 1e-8))
    
    # 4. Composite NEXUS Risk Algorithm
    risk = 0.0
    # Class-based severity
    if label == "Glioma": risk += 45
    elif label == "Meningioma": risk += 25
    elif label == "Pituitary": risk += 15
        
    # Weighted modifiers
    risk += confidence * 15
    risk += min(activation_area * 150, 20) 
    risk += normalized_entropy * 20
    
    final_risk = min(max(int(risk), 10), 99)
    
    return {
        "entropy": round(float(ent_val), 2),
        "irregularity_ratio": round(float(normalized_entropy), 2),
        "activation_area": round(float(activation_area), 3),
        "asymmetry": round(asymmetry, 3),
        "risk_score": final_risk
    }

def numpy_to_base64(img_np):
    """Encodes clinical images for HUD display."""
    pil_img = Image.fromarray(img_np.astype('uint8'))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
