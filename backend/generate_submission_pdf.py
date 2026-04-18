from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

class NeuralNexusPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 20)
        self.set_text_color(0, 200, 255)
        self.cell(0, 10, 'NEURAL NEXUS', align='L', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font('helvetica', 'I', 10)
        self.set_text_color(128, 128, 128)
        self.set_y(10)
        self.cell(0, 10, 'Clinical AI Project Submission', align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(5)
        self.set_draw_color(0, 200, 255)
        self.line(10, 22, 200, 22)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_report():
    pdf = NeuralNexusPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Page
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font('helvetica', 'B', 28)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(190, 15, 'Neural Nexus:\nUnified AI Brain Tumor Diagnostic System', align='C')
    pdf.ln(15)
    pdf.set_font('helvetica', '', 14)
    pdf.multi_cell(190, 10, 'Technical Project Report & Submission Documentation', align='C')
    
    # Large Placeholder on front page
    pdf.set_y(160)
    pdf.set_fill_color(245, 250, 255)
    pdf.set_draw_color(200, 200, 200)
    pdf.rect(20, 160, 170, 70, 'DF')
    pdf.set_xy(20, 190)
    pdf.set_font('helvetica', 'I', 12)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(170, 10, '[ INSERT PROJECT LOGO / CLINICAL HUD PREVIEW HERE ]', align='C')
    
    slides = [
        ("The Clinical Challenge", [
            "Addressing the 'Black Box' problem in AI diagnostics.",
            "Consolidating fragmented radiological workflows.",
            "Quantifying pathological risks in real-time."
        ]),
        ("ML Core Architecture", [
            "Model: ResNet-50 optimized for neuroimaging.",
            "Classification: Glioma, Meningioma, Pituitary, Normal.",
            "Technology: PyTorch, FastAPI, HuggingFace Inference."
        ]),
        ("Explainable AI (XAI) Engine", [
            "Method: Custom Grad-CAM activation mapping.",
            "Interactive HUD: Real-time intensity adjustments.",
            "Clinical Relevance: Pinpointing focus regions for transparent verification."
        ]),
        ("BioMistral Integration", [
            "Model: BioMistral-7B (SOTA Medical LLM).",
            "Output: Natural language explanations of AI predictions.",
            "Condition Analysis: Structured risk assessment and recommendations."
        ]),
        ("Quantitative Risk Metrics", [
            "Boundary Entropy: Mathematical irregularity measurement.",
            "Activation Area: Involvement percentage of brain volume.",
            "Integrated Risk Score: Composite metric for triage prioritization."
        ])
    ]

    for title, points in slides:
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 18)
        pdf.set_text_color(0, 100, 200)
        pdf.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(5)
        
        pdf.set_font('helvetica', '', 12)
        pdf.set_text_color(0, 0, 0)
        for point in points:
            # Explicitly set X to margin to avoid "not enough horizontal space" error
            pdf.set_x(10)
            pdf.multi_cell(190, 10, f'- {point}')
        
        # Add screenshot placeholder
        pdf.ln(10)
        current_y = pdf.get_y()
        # If space is tight, move to next page
        if current_y > 200:
            pdf.add_page()
            current_y = pdf.get_y()
            
        pdf.set_fill_color(252, 252, 252)
        pdf.rect(10, current_y, 190, 80, 'DF')
        pdf.set_xy(10, current_y + 35)
        pdf.set_font('helvetica', 'BI', 10)
        pdf.set_text_color(180, 180, 180)
        pdf.cell(190, 10, '[ INSERT RELEVANT CLINICAL SCREENSHOT HERE ]', align='C')

    # Final Conclusion
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 18)
    pdf.set_text_color(0, 100, 200)
    pdf.cell(0, 10, 'Conclusion & Future Outlook', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(5)
    pdf.set_font('helvetica', '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(190, 10, "Neural Nexus represents a significant step towards transparent and efficient AI assistance in neuro-oncology. By combining advanced classification with XAI and medical LLM narratives, we provide a unified platform that empowers clinicians rather than replacing them.")
    
    # Extra space for final screenshots
    pdf.ln(10)
    pdf.set_fill_color(252, 252, 252)
    pdf.rect(10, pdf.get_y(), 190, 80, 'DF')
    pdf.set_y(pdf.get_y() + 35)
    pdf.cell(190, 10, '[ INSERT FINAL SYSTEM WALKTHROUGH IMAGE HERE ]', align='C')

    output_path = "Neural_Nexus_Project_Submission.pdf"
    pdf.output(output_path)
    print(f"PDF successfully generated: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_report()
