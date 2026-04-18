# Neural-NEXUS: Clinical-Grade Brain Tumor Diagnostics via Interpretable Deep Learning 

Neural-NEXUS is a specialized deep learning framework engineered for the robust analysis and classification of brain tumors from MRI imaging. This repository characterizes a system designed to navigate fundamental challenges in medical AI: dataset class disparity, inter-patient variability, and the critical requirement for clinical interpretability.

---

## Clinical Context and Objectives

The integration of AI into clinical workflows demands a shift from "Black-Box" models toward interpretable, mathematically sound systems. Neural-NEXUS was developed to address four primary theoretical hurdles in automated MRI analysis:

1.  **Metric Reliability**: Ensuring rare tumor types are not marginalized by majority classes.
2.  **Morphological Generalization**: Learning pathologically relevant features rather than patient-specific noise.
3.  **The Interpretability Gap**: Providing visual evidence to bridge the trust-gap between AI and clinical professionals.
4.  **Domain Adaptation**: Leveraging generalized feature knowledge for specific medical morphologies.

---

## Theoretical Foundations

### 1. Residual Learning and Identity Mapping
Neural-NEXUS utilizes a ResNet50 backbone, which is theoretically grounded in the solution to the **degradation problem** observed in deep networks. As models increase in depth, accuracy often saturates and then degrades rapidly.

*   **The Residual Solution**: Instead of hoping each stack of layers learns a direct mapping $H(x)$, we cast the layers to fit a residual mapping $F(x) = H(x) - x$. The original mapping is then recast into $F(x) + x$.
*   **The Mathematical Advantage**: It is significantly easier to optimize a residual mapping than a full mapping. If an identity mapping is optimal, the network can easily drive the weights to zero, effectively utilizing "skip-connections" to pass information forward without the risk of vanishing or exploding gradients.

### 2. Weighted Cross-Entropy and Gradient Scaling
In medical datasets, class imbalance is a natural byproduct of pathology frequency. Without intervention, the model develops a mathematical bias toward more common categories (e.g., Gliomas).

*   **Inverse-Frequency Weighting**: We adjust the Cross-Entropy loss function by a weight vector $W$. The penalty for a specific class $j$ is scaled by:
    $w_j = \frac{N}{C \cdot n_j}$
    where $N$ is the total samples, $C$ is the number of classes, and $n_j$ is the count for class $j$.
*   **Theoretical Impact**: This scaling ensures that the gradient magnitude from rare classes is amplified, forcing the optimizer to treat every tumor type with equal diagnostic priority during weight updates.

### 3. Grad-CAM: Gradient-Weighted Class Activation Mapping
Clinical transparency is achieved via Grad-CAM, which produces a localization map $L^c_{Grad-CAM}$ for a given class $c$.

*   **Derivative Projection**: We compute the gradient of the score for class $c$, $y^c$, with respect to feature map activations $A^k$ of a convolutional layer.
*   **Weight Computation**: These gradients are global-average-pooled to obtain importance weights $\alpha^c_k$:
    $\alpha^c_k = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}$
*   **Linear Combination**: The final heatmap is a ReLU-activated weighted sum of the feature maps:
    $L^c_{Grad-CAM} = ReLU(\sum_k \alpha^c_k A^k)$
*   **Clinical Significance**: This process identifies the specific structural features (e.g., contrast anomalies or density shifts) that the model identified as pathological.

### 4. Transfer Learning and Domain Adaptation
We leverage **Feature Representation Learning** from the ImageNet domain. Theoretical research suggests that early convolutional layers learn "general" attributes (edges, textures) that are universal across image domains. By fine-tuning the deep "semantic" layers, we adapt these generalized filters to the specific structural morphology of neurological MRI.

---

## Technical Specifications

### Architecture: ResNet50
*   **Depth**: 50 layers of residual learning.
*   **Activations**: SiLU (Sigmoid Linear Unit/Swish). Unlike ReLU, SiLU is a non-monotonic, smooth activation function that allows for better gradient flow and reduces the risk of "dead" neurons in high-stakes diagnostic environments.
*   **Regularization**: High-ratio (0.5) Stochastic Dropout. This prevents the model from developing codependency between neurons, ensuring it learns independent features that generalize to new patients.

### Optimization Strategy
*   **Optimizer**: AdamW (Adaptive Moment Estimation with Decoupled Weight Decay).
*   **Scheduler**: ReduceLROnPlateau, which simulates an annealing process, sharpening the model's focus as the test-error stabilizes.

---

## Dataset Characteristics

**Kaggle Dataset Source**: [Brain Tumor Healthcare Dataset](https://www.kaggle.com/datasets/purvanshjoshi1/healthcare)

| Pathology Category | Image Count | Theory of Role |
| :--- | :--- | :--- |
| **Glioma** | 5,625 | High-volume positive class |
| **Meningioma** | 3,978 | Structural positive class |
| **Pituitary** | 4,363 | Endocrine-origin positive class |
| **Healthy Control (No Tumor)** | 3,847 | Baseline / Negative Control |

---

## Performance Analysis & Clinical Validation

### Confusion Matrix Evaluation
The system demonstrated a **91.88% confusion matrix accuracy**, proof of its theoretical stability across all categories.

<img width="1392" height="1046" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/3f6a8497-37ef-4abd-9422-00cd68ef4604" />

### Interpretability Gallery (Grad-CAM Results)

#### Tumor Localization Theory
<img width="1013" height="492" alt="Tumor Localization" src="https://github.com/user-attachments/assets/6d22032e-b411-4c6c-a83d-e0a391909178" />
*Visualization of localized anomalies; the heatmaps correspond to specific density variations identified by the residual blocks.*

#### Control Case Verification
<img width="895" height="461" alt="Healthy Part" src="https://github.com/user-attachments/assets/f49be335-3013-437c-a29b-6b7121c7c2f4" />
*In "No Tumor" control cases, the attention distribution confirms the model is evaluating structural symmetry rather than background noise.*

#### Extended Diagnostic Report
<img width="1768" height="926" alt="Full Report" src="https://github.com/user-attachments/assets/d8a946fb-fb62-4c15-8086-2b481171e6fa" />
<img width="878" height="464" alt="Prediction 4" src="https://github.com/user-attachments/assets/b882cbe1-7e8a-4683-9b17-6418027beda7" />

#### Summary
<img width="1730" height="952" alt="image" src="https://github.com/user-attachments/assets/6205b700-5f1b-4b85-9e4c-60a5cd0d6702" />


---
##	System Architecture Overview

	Neural Nexus is a state-of-the-art AI diagnostic platform designed to assist clinicians in the detection, classification, and reporting of brain tumors from MRI scans. The system follows a decoupled architecture with a high-performance Python backend and a cinematic, interactive React frontend.
	1. AI Diagnostic Core (Backend)Engine: Built with FastAPI, the backend orchestrates the diagnostic pipeline.
	Model Architecture: Uses a custom-trained EfficientNet-V2-S / ResNet-50 CNN, optimized for high-accuracy classification across four primary classes: Glioma, Meningioma, Pituitary, and No Tumor.
	Interpretability (Grad-CAM): Implements Gradient-weighted Class Activation Mapping to generate visual heatmaps, highlighting the specific regions in the MRI scan that drove the AI's classification decision.
	Clinical Narrative (BioMistral): Integrates the BioMistral LLM to transform raw diagnostic telemetry and risk metrics into human-readable clinical narratives and risk assessments.
	Reporting: Automatically generates dynamic, clinician-ready PDF reports using fpdf2.
	2. Clinical HUD & Visualization (Frontend)Framework: Developed using React and Vite, prioritizing a high-density, "Mission Control" aesthetic.
	Cross-Modality Viewer: Supports multiple diagnostic views:RAW: The original MRI input.ENHANCED: CLAHE-processed image for better contrast.
	GRAD-CAM: Heatmap overlay at varying intensities.
	SPLIT-VIEW: Interactive wipe separator for side-by-side comparison
	Spatial Copilot (3D Brain): An interactive 3D model powered by React Three Fiber (Three.js). It maps the 2D tumor location from the scan into a 3D coordinate system, providing anatomical context for the detection.
	Dynamic Telemetry: Real-time HUD displaying confidence scores, classification results, and clinical risk assessments.
	 Deployment & Scalability
Frontend:https://neural-nexus-git-feature-un-e2e0e7-purvansh-s-projects-3c13a221.vercel.app/
Backed :https://huggingface.co/spaces/purvansh01/neural-nexus-backend
	Infrastructure: The backend is designed for containerized deployment (e.g., Hugging Face Spaces via Docker), while the frontend is optimized for edge deployment (e.g., Vercel).
