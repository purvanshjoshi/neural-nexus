from fpdf import FPDF
import base64
import io
from PIL import Image
from datetime import datetime
import tempfile
import os

class NeuralNexusReport(FPDF):
    def header(self):
        # Header with branding
        self.set_fill_color(21, 25, 28) # Dark Slate like the UI
        self.rect(0, 0, 210, 40, 'F')
        
        self.set_font('helvetica', 'B', 24)
        self.set_text_color(255, 255, 255)
        self.cell(0, 20, 'NEURAL NEXUS', ln=True, align='L')
        
        self.set_font('helvetica', '', 10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 0, 'Official AI Clinical Diagnostic Report', ln=True, align='L')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Confidential Medical Document', align='C')

def generate_clinical_report(data):
    """
    Generates a PDF report from analysis data.
    'data' should contain: label, confidence, probabilities, and images dict.
    """
    pdf = NeuralNexusReport()
    pdf.add_page()
    
    # 1. Summary Section
    pdf.set_font('helvetica', 'B', 16)
    pdf.set_text_color(0, 102, 255) # Accent Blue
    pdf.cell(0, 10, f"PRIMARY DIAGNOSIS: {data['label']}", ln=True)
    
    pdf.set_font('helvetica', '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, f"System Confidence: {data['confidence']:.2%}", ln=True)
    pdf.ln(5)
    
    # 2. Visualizations
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(95, 10, 'Original MRI Frame', align='C')
    pdf.cell(95, 10, 'AI Attention Map (Grad-CAM)', ln=True, align='C')
    
    # Position for images
    img_x = 10
    img_y = pdf.get_y()

    def add_base64_img(b64_str, x, y, width=90):
        try:
            img_data = base64.b64decode(b64_str)
            # Use PIL to normalize the image to RGB JPEG (bypasses buggy fpdf2 PNG parser)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # Save to a temp JPEG
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            with os.fdopen(fd, 'wb') as tmp:
                img.save(tmp, format="JPEG", quality=95)
            
            # Embed in PDF
            pdf.image(tmp_path, x=x, y=y, w=width)
            
            # Clean up immediately
            os.remove(tmp_path)
        except Exception as e:
            print(f"Failed to process image for PDF: {e}")
            pdf.set_xy(x, y)
            pdf.cell(width, 10, "[Image Ingestion Failed]", border=1, align='C')
        
    add_base64_img(data['images']['original'], img_x, img_y)
    add_base64_img(data['images']['heatmap'], img_x + 100, img_y)
    
    pdf.set_y(img_y + 95) # Move below images
    pdf.ln(10)
    
    # 3. Probability Breakdown
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'Probability Distribution Breakdown', ln=True)
    pdf.ln(2)
    
    pdf.set_font('helvetica', '', 10)
    for cls, prob in data['probabilities'].items():
        # Label
        pdf.cell(40, 8, f"{cls}:", align='L')
        # Value
        pdf.cell(20, 8, f"{prob:.2%}", align='R')
        # Simple Bar representation
        pdf.set_fill_color(200, 200, 200)
        pdf.rect(75, pdf.get_y() + 2, 100, 4, 'F')
        pdf.set_fill_color(0, 102, 255)
        pdf.rect(75, pdf.get_y() + 2, 100 * prob, 4, 'F')
        pdf.ln(8)
    
    # 4. Uncertainty Analysis Section (Optional - only if MC Dropout data present)
    if data.get('reliability_score') is not None:
        pdf.add_page()
        
        # Section Header
        pdf.set_font('helvetica', 'B', 16)
        pdf.set_text_color(0, 102, 255)
        pdf.cell(0, 10, 'UNCERTAINTY ANALYSIS (Monte Carlo Dropout)', ln=True)
        pdf.ln(5)
        
        pdf.set_font('helvetica', '', 10)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 5, f"This analysis was performed using {data.get('n_samples', 30)} stochastic forward passes with dropout active during inference. This Bayesian approximation technique provides clinically meaningful uncertainty estimates beyond standard softmax confidence.")
        pdf.ln(8)
        
        # Reliability Score
        reliability = data.get('reliability_score', 0)
        if reliability >= 90:
            reliability_label = "VERY HIGH"
            r, g, b = 0, 180, 80
        elif reliability >= 70:
            reliability_label = "HIGH"
            r, g, b = 0, 160, 200
        elif reliability >= 50:
            reliability_label = "MODERATE"
            r, g, b = 255, 180, 0
        else:
            reliability_label = "LOW - SECOND OPINION RECOMMENDED"
            r, g, b = 255, 50, 50
        
        pdf.set_font('helvetica', 'B', 14)
        pdf.set_text_color(r, g, b)
        pdf.cell(0, 10, f"Clinical Reliability Score: {reliability}/100 ({reliability_label})", ln=True)
        pdf.ln(5)

        # Uncertainty Metrics Table
        pdf.set_font('helvetica', 'B', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, 'Key Uncertainty Metrics', ln=True)
        pdf.ln(2)
        
        pdf.set_font('helvetica', '', 10)

        metrics = [
            ("Mean Confidence", f"{data.get('mean_confidence', 0):.2%}"),
            ("Confidence Std Dev", f"{data.get('std_confidence', 0):.4f}"),
            ("Predictive Entropy", f"{data.get('entropy', 0):.4f}"),
            ("Mutual Information (Epistemic)", f"{data.get('mutual_information', 0):.4f}"),
        ]
        
        for label, value in metrics:
            pdf.cell(80, 7, f"  {label}:", align='L')
            pdf.cell(60, 7, value, ln=True, align='L')
        
        pdf.ln(5)

        # Per-class variance breakdown
        std_probs = data.get('std_probs', {})
        if std_probs:
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 8, 'Per-Class Prediction Variance', ln=True)
            pdf.ln(2)
            
            pdf.set_font('helvetica', '', 10)
            for cls, std_val in std_probs.items():
                pdf.cell(40, 7, f"  {cls}:", align='L')
                pdf.cell(25, 7, f"\u00b1{std_val:.4f}", align='R')
                # Variance bar
                pdf.set_fill_color(220, 220, 220)
                pdf.rect(80, pdf.get_y() + 2, 100, 3, 'F')
                bar_width = min(std_val * 400, 100)  # Scale for visualization
                pdf.set_fill_color(255, 80, 80)
                pdf.rect(80, pdf.get_y() + 2, bar_width, 3, 'F')
                pdf.ln(7)
        
        pdf.ln(5)

        # Prediction Consistency
        pred_counts = data.get('prediction_counts', {})
        n_samples = data.get('n_samples', 30)
        if pred_counts:
            pdf.set_font('helvetica', 'B', 11)
            pdf.cell(0, 8, f'Prediction Consistency Across {n_samples} MC Samples', ln=True)
            pdf.ln(2)
            
            pdf.set_font('helvetica', '', 10)
            for cls, count in pred_counts.items():
                pct = (count / n_samples) * 100
                pdf.cell(40, 7, f"  {cls}:", align='L')
                pdf.cell(25, 7, f"{count}/{n_samples} ({pct:.0f}%)", align='R')
                # Consistency bar
                pdf.set_fill_color(220, 220, 220)
                pdf.rect(80, pdf.get_y() + 2, 100, 3, 'F')
                pdf.set_fill_color(0, 102, 255)
                pdf.rect(80, pdf.get_y() + 2, pct, 3, 'F')
                pdf.ln(7)
        
        pdf.ln(5)
        
        # Methodology note
        pdf.set_fill_color(240, 248, 255)
        pdf.rect(10, pdf.get_y(), 190, 22, 'F')
        pdf.set_font('helvetica', 'I', 8)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 4, "   Methodology: Monte Carlo Dropout (Gal & Ghahramani, ICML 2016) approximates Bayesian inference by\n   sampling from the posterior predictive distribution. Higher reliability scores indicate consistent model behavior\n   across stochastic perturbations. Low scores suggest the prediction lies near a decision boundary.")
    
    # 5. Final Disclaimer
    pdf.ln(10)
    pdf.set_fill_color(245, 245, 245)
    pdf.rect(10, pdf.get_y(), 190, 20, 'F')
    pdf.set_font('helvetica', 'B', 8)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "   CLINICAL DISCLAIMER:", ln=True)
    pdf.set_font('helvetica', '', 8)
    pdf.multi_cell(0, 4, "   Neural Nexus is an AI-assisted diagnostic tool. This report is intended for use by qualified medical professionals only. \n   Final diagnosis must be confirmed through professional clinical correlation.")

    # Return as bytes
    return pdf.output()
