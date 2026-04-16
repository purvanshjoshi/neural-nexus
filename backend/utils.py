import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import io
import base64
from scipy.stats import entropy

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

    def generate(self, input_tensor, class_idx=None, mc_samples=0):
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
    Score-CAM: Gradient-free visualization using model scores.
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
        
        # Normalize each channel's weight
        max_val = upsampled.view(upsampled.size(0), upsampled.size(1), -1).max(dim=2)[0].view(upsampled.size(0), upsampled.size(1), 1, 1)
        min_val = upsampled.view(upsampled.size(0), upsampled.size(1), -1).min(dim=2)[0].view(upsampled.size(0), upsampled.size(1), 1, 1)
        upsampled = (upsampled - min_val) / (max_val - min_val + 1e-8)
        
        # Mask input with each channel
        masked_input = input_tensor * upsampled
        
        # Get scores for masked inputs
        # To avoid memory issues with large C, we do it in batches if needed, 
        # but here we'll assume a reasonable C (e.g. 512 or 2048)
        # For ResNet50 layer4, C=2048. Let's do a simplified version.
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

def calculate_risk_metrics(heatmap):
    """
    Extracts quantitative clinical risk features from the heatmap.
    """
    # 1. Heatmap Entropy (Irregularity/Chaos)
    # Flatten and normalize to a distribution
    h_flat = heatmap.flatten()
    h_dist = h_flat / (np.sum(h_flat) + 1e-8)
    h_entropy = float(entropy(h_dist))
    
    # 2. Activation Area Ratio (Tumor size approximation)
    # Threshold at 0.5 to find "high focus" zones
    high_focus = (heatmap > 0.5).astype(np.float32)
    area_ratio = float(np.sum(high_focus) / heatmap.size)
    
    # 3. Hemispheric Asymmetry
    # Split heatmap into left and right halves
    width = heatmap.shape[1]
    left_half = heatmap[:, :width//2]
    right_half = heatmap[:, width//2:]
    
    left_mass = np.sum(left_half)
    right_mass = np.sum(right_half)
    asymmetry = float(abs(left_mass - right_mass) / (left_mass + right_mass + 1e-8))
    
    # 4. NEXUS Risk Score (Composite 1-100)
    # Weights: Area (40%), Entropy (40%), Asymmetry (20%)
    # Normalized components
    norm_area = min(area_ratio / 0.1, 1.0) # 10% brain coverage is "max risk" for area
    norm_entropy = min(h_entropy / 10.0, 1.0) # Entropy usually scales 0-10+
    
    risk_score = (norm_area * 0.4 + norm_entropy * 0.4 + asymmetry * 0.2) * 100
    
    return {
        "entropy": h_entropy,
        "area_ratio": area_ratio,
        "asymmetry": asymmetry,
        "risk_score": round(risk_score, 1)
    }

def numpy_to_base64(img_np):
    """Converts numpy image (RGB) to Base64 string for UI display"""
    pil_img = Image.fromarray(img_np.astype('uint8'))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
