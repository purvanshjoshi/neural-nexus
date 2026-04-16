"""
Monte Carlo Dropout Uncertainty Quantification Engine
=====================================================
Implements Bayesian uncertainty estimation via stochastic forward passes.
By keeping dropout active during inference, we approximate the posterior
predictive distribution and derive clinically meaningful uncertainty metrics.

References:
    Gal & Ghahramani, "Dropout as a Bayesian Approximation" (ICML 2016)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Enables dropout layers during inference for Monte Carlo sampling.
    Only activates nn.Dropout modules; leaves BatchNorm in eval mode.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def disable_mc_dropout(model: nn.Module) -> None:
    """
    Restores all modules back to eval mode after MC sampling.
    """
    model.eval()


def mc_dropout_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_samples: int = 30,
    classes: List[str] = None,
) -> Dict:
    """
    Performs N stochastic forward passes with dropout active.

    Returns:
        dict with keys:
            - mean_probs: averaged softmax probabilities per class
            - std_probs: standard deviation per class (epistemic uncertainty)
            - predicted_class_idx: index of the most probable class
            - mean_confidence: mean confidence for the predicted class
            - std_confidence: std of confidence for the predicted class
            - entropy: predictive entropy (total uncertainty)
            - mutual_information: epistemic uncertainty via mutual info
            - reliability_score: clinical reliability metric (0-100)
            - all_predictions: list of all N prediction arrays
            - prediction_counts: how many times each class was predicted as top-1
    """
    if classes is None:
        classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    model.eval()
    enable_mc_dropout(model)

    all_probs = []

    with torch.no_grad():
        for _ in range(n_samples):
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy()[0])

    disable_mc_dropout(model)

    all_probs = np.array(all_probs)  # Shape: (n_samples, n_classes)

    # --- Core Statistics ---
    mean_probs = np.mean(all_probs, axis=0)
    std_probs = np.std(all_probs, axis=0)

    predicted_class_idx = int(np.argmax(mean_probs))
    mean_confidence = float(mean_probs[predicted_class_idx])
    std_confidence = float(std_probs[predicted_class_idx])

    # --- Information-Theoretic Metrics ---
    # Predictive Entropy: H[y|x,D] = -sum(mean_p * log(mean_p))
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-10)))

    # Expected Entropy: E_theta[H[y|x,theta]]
    expected_entropy = float(
        -np.mean(np.sum(all_probs * np.log(all_probs + 1e-10), axis=1))
    )

    # Mutual Information (epistemic uncertainty) = Predictive Entropy - Expected Entropy
    mutual_information = float(entropy - expected_entropy)

    # --- Clinical Reliability Score (0-100) ---
    reliability_score = compute_reliability_score(mean_confidence, std_confidence, entropy)

    # --- Top-1 prediction counts across samples ---
    top1_preds = np.argmax(all_probs, axis=1)
    prediction_counts = {
        classes[i]: int(np.sum(top1_preds == i)) for i in range(len(classes))
    }

    return {
        "mean_probs": {classes[i]: float(mean_probs[i]) for i in range(len(classes))},
        "std_probs": {classes[i]: float(std_probs[i]) for i in range(len(classes))},
        "predicted_class_idx": predicted_class_idx,
        "predicted_label": classes[predicted_class_idx],
        "mean_confidence": mean_confidence,
        "std_confidence": std_confidence,
        "entropy": entropy,
        "mutual_information": mutual_information,
        "reliability_score": reliability_score,
        "all_predictions": all_probs.tolist(),
        "prediction_counts": prediction_counts,
        "n_samples": n_samples,
    }


def compute_reliability_score(
    mean_conf: float, std_conf: float, entropy: float
) -> float:
    """
    Computes a composite clinical reliability score (0-100).

    Components:
        - Confidence component (40%): Higher mean confidence = more reliable
        - Consistency component (35%): Lower std = more consistent predictions
        - Entropy component (25%): Lower entropy = less total uncertainty

    The score is designed so that:
        - 90-100: Very High reliability (green)
        - 70-89:  High reliability (cyan)
        - 50-69:  Moderate reliability (amber)
        - 0-49:   Low reliability (red) — consider second opinion
    """
    # Confidence component: direct mapping (0-1 -> 0-40)
    conf_score = mean_conf * 40.0

    # Consistency component: low std is good (0 std -> 35, high std -> 0)
    # Max std for 4-class softmax is ~0.5
    consistency_score = max(0.0, (1.0 - std_conf * 4.0)) * 35.0

    # Entropy component: low entropy is good
    # Max entropy for 4 classes = ln(4) ≈ 1.386
    max_entropy = np.log(4)
    entropy_score = max(0.0, (1.0 - entropy / max_entropy)) * 25.0

    total = conf_score + consistency_score + entropy_score
    return round(min(100.0, max(0.0, total)), 1)


def generate_uncertainty_heatmap(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    n_samples: int = 15,
) -> np.ndarray:
    """
    Generates a spatial uncertainty map by running MC Dropout on Grad-CAM.

    For each stochastic forward pass, we compute a Grad-CAM heatmap.
    The pixel-wise standard deviation across all heatmaps reveals
    WHERE the model is uncertain — complementary to standard Grad-CAM
    which shows where the model LOOKS.

    Returns:
        uncertainty_heatmap: normalized numpy array (0-1)
    """
    model.eval()
    enable_mc_dropout(model)

    all_heatmaps = []
    gradients_store = [None]
    activations_store = [None]

    # Register hooks
    def save_activation(module, input, output):
        activations_store[0] = output.detach()

    def save_gradient(module, grad_input, grad_output):
        gradients_store[0] = grad_output[0].detach()

    hook_fwd = target_layer.register_forward_hook(save_activation)
    hook_bwd = target_layer.register_full_backward_hook(save_gradient)

    try:
        for _ in range(n_samples):
            enable_mc_dropout(model)

            input_clone = input_tensor.clone().requires_grad_(True)
            output = model(input_clone)
            class_idx = output.argmax(dim=1).item()

            model.zero_grad()
            output[0, class_idx].backward(retain_graph=False)

            if gradients_store[0] is not None and activations_store[0] is not None:
                pooled_grads = torch.mean(gradients_store[0], dim=[0, 2, 3])
                acts = activations_store[0].clone()

                for i in range(acts.shape[1]):
                    acts[:, i, :, :] *= pooled_grads[i]

                heatmap = torch.mean(acts, dim=1).squeeze()
                heatmap = np.maximum(heatmap.cpu().numpy(), 0)
                heatmap_max = np.max(heatmap)
                if heatmap_max > 0:
                    heatmap /= heatmap_max
                all_heatmaps.append(heatmap)
    finally:
        hook_fwd.remove()
        hook_bwd.remove()
        disable_mc_dropout(model)

    if len(all_heatmaps) < 2:
        # Fallback: return zero heatmap if MC sampling failed
        return np.zeros((7, 7), dtype=np.float32)

    # Stack and compute pixel-wise standard deviation
    heatmap_stack = np.stack(all_heatmaps, axis=0)
    uncertainty_map = np.std(heatmap_stack, axis=0)

    # Normalize to 0-1
    unc_max = np.max(uncertainty_map)
    if unc_max > 0:
        uncertainty_map /= unc_max

    return uncertainty_map.astype(np.float32)
