import os
from huggingface_hub import InferenceClient
from typing import Dict, Optional

# Initialize the HF Inference Client
# It will look for HF_TOKEN in environment variables
HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient("BioMistral/BioMistral-7B", token=HF_TOKEN)

def generate_clinical_narrative(analysis_data: Dict, patient_meta: Optional[Dict] = None) -> str:
    """
    Takes ResNet-50/EfficientNet output + Grad-CAM metadata and generates
    a radiologist-style clinical narrative via BioMistral.
    """
    
    label = analysis_data.get('label', 'Unknown')
    confidence = analysis_data.get('confidence', 0.0)
    probabilities = analysis_data.get('probabilities', {})
    
    # Format probabilities for the prompt
    prob_str = ", ".join([f"{k}: {v:.1%}" for k, v in probabilities.items()])
    
    # Estimate hemisphere based on tumor location if available
    loc = analysis_data.get('tumor_location')
    hemisphere = "undetermined"
    if loc:
        hemisphere = "Right hemisphere" if loc.get('x', 0.5) > 0.5 else "Left hemisphere"

    prompt = f"""<s>[INST] You are a board-certified neuroradiologist reviewing an AI-assisted brain MRI analysis. 
Generate a structured clinical impression based on the following AI data.

ANALYSIS DATA:
- Primary Diagnosis: {label}
- Confidence: {confidence:.1%}
- Probability Distribution: {prob_str}
- Estimated Lateralization: {hemisphere}

{f"PATIENT CONTEXT: {patient_meta}" if patient_meta else ""}

Generate a report with the following sections:
1. RADIOLOGICAL IMPRESSION: (2-3 sentences describing the finding and its implications)
2. DIFFERENTIAL CONSIDERATIONS: (List 1-2 other possibilities if appropriate)
3. RECOMMENDED FOLLOW-UP: (Immediate next steps in clinical workflow)
4. CLINICAL CAVEATS: (Standard AI disclaimer)

Use formal medical terminology. Do NOT provide a definitive diagnosis—frame as AI-assisted findings requiring clinical correlation. [/INST]"""

    try:
        response = client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.3, # Low creativity for medical accuracy
            repetition_penalty=1.1,
            do_sample=True
        )
        return response.strip()
    except Exception as e:
        return f"Clinical Narrative Generation Unavailable: {str(e)}"
