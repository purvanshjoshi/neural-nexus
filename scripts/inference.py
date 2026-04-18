import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-V2.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations by pooled gradients
        activations = self.activations.detach().clone()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= np.max(heatmap) + 1e-8
        
        # Calculate confidence
        probs = torch.softmax(output, dim=1)
        confidence = probs[0, class_idx].item()
        
        all_probs = {f"Class {i}": float(p) for i, p in enumerate(probs[0])}
        
        return heatmap, class_idx, confidence, all_probs

class HybridEfficientNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.efficientnet_v2_s(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return self.head(features)

def load_model(weights_path, num_classes=4, device='cpu'):
    model = HybridEfficientNet(num_classes=num_classes)
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading weights: {e}")
        # Return a fresh model if weights fail to load for local testing
    model.to(device)
    model.eval()
    return model

def preprocess_image(image, img_size=256, use_clahe=True):
    """
    Accepts a PIL Image or NumPy array and returns a preprocessed tensor.
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert RGB to BGR for OpenCV if needed, then back to RGB
    # But Gradio gives RGB, so we stay in RGB
    
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = np.stack([clahe.apply(image[:,:,i]) for i in range(3)], axis=-1)
    
    processed_img_for_display = image.copy()
    
    # Resize and Normalize
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, processed_img_for_display

def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # Convert back to RGB for display
    overlayed_img = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlayed_img
