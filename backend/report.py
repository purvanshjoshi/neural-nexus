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
    
    # 1.5 Clinical Narrative
    if data.get('clinical_narrative'):
        pdf.set_font('helvetica', 'B', 12)
        pdf.cell(0, 8, "AI-Assisted Clinical Narrative", ln=True)
        pdf.set_font('helvetica', '', 10)
        pdf.set_text_color(50, 50, 50)
        # Use latin-1 compatible characters or let fpdf2 handle Unicode
        narrative = data['clinical_narrative']
        pdf.multi_cell(0, 5, narrative)
        pdf.ln(5)
        pdf.set_text_color(0, 0, 0)

    
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
    
    # 3. Risk Stratification Section
    if 'risk_metrics' in data and data['risk_metrics']:
        metrics = data['risk_metrics']
        pdf.ln(10)
        pdf.set_font('helvetica', 'B', 12)
        pdf.set_text_color(255, 100, 0) # Risk Orange
        pdf.cell(0, 10, f"NEXUS RISK ASSESSMENT | SCORE: {metrics.get('risk_score', 'N/A')}/100", ln=True)
        
        pdf.set_font('helvetica', '', 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, f"- Heatmap Entropy (Chaos): {metrics.get('entropy', 0):.4f}", ln=True)
        pdf.cell(0, 7, f"- Activation Area Ratio: {metrics.get('area_ratio', 0):.2%}", ln=True)
        pdf.cell(0, 7, f"- Hemispheric Asymmetry: {metrics.get('asymmetry', 0):.2%}", ln=True)

    # 4. Clinical Narrative Section (BioMistral)
    if 'narrative' in data and data['narrative']:
        pdf.add_page() # Put narrative on a fresh page if it's long
        pdf.set_font('helvetica', 'B', 14)
        pdf.set_text_color(0, 102, 255)
        pdf.cell(0, 10, "RADIOLOGICAL IMPRESSION (AI-GENERATED)", ln=True)
        pdf.ln(5)
        
        pdf.set_font('helvetica', '', 11)
        pdf.set_text_color(40, 40, 40)
        # Multi-cell for long text
        pdf.multi_cell(0, 6, data['narrative'])
        pdf.ln(10)

    # 5. Probability Breakdown
    pdf.set_y(pdf.get_y() + 5)
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(0, 0, 0)
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
    
    # 6. Final Disclaimer
    pdf.ln(10)
    pdf.set_fill_color(245, 245, 245)
    # Check if we have enough room on page
    if pdf.get_y() > 250:
        pdf.add_page()
    
    pdf.rect(10, pdf.get_y(), 190, 25, 'F')
    pdf.set_font('helvetica', 'B', 8)
    pdf.cell(0, 8, "   CLINICAL DISCLAIMER:", ln=True)
    pdf.set_font('helvetica', '', 8)
    pdf.multi_cell(0, 4, "   Neural Nexus is an AI-assisted diagnostic tool. This report is intended for use by qualified medical professionals only. \n   BioMistral narratives are research-only and must be clinically correlated. Final diagnosis must be confirmed through professional clinical correlation.")

    # Return as bytes
    return pdf.output()
