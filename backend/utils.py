import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
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
