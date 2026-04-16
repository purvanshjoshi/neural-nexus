import os
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. MODELS & CLIENT INITIALIZATION
# ==========================================

# BioMistral-7B: Specialized for clinical narratives.
# Mistral-7B-v0.2: A robust general-purpose fallback.
PRIMARY_MODEL = "BioMistral/BioMistral-7B"
FALLBACK_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

hf_token = os.getenv("HF_TOKEN")
client = InferenceClient(model=PRIMARY_MODEL, token=hf_token)
fallback_client = InferenceClient(model=FALLBACK_MODEL, token=hf_token)

def _generate_text(prompt: str, max_new_tokens: int = 512, temperature: float = 0.3) -> str:
    """Helper to generate text with fallback logic using chat_completion."""
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = client.chat_completion(
            messages,
            max_tokens=max_new_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"BioMistral inference failed ({e}). Falling back to {FALLBACK_MODEL}...")
        try:
            response = fallback_client.chat_completion(
                messages,
                max_tokens=max_new_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as fallback_e:
            print(f"Fallback inference failed: {fallback_e}")
            return "Error: Unable to generate response due to API limits or model unavailability."

# ==========================================
# 2. CORE NARRATIVE GENERATOR
# ==========================================

def generate_clinical_narrative(analysis_data: dict, risk_metrics: dict, patient_meta: dict = None) -> str:
    """
    Takes model outputs and risk metrics to generate a professional radiologist-style narrative.
    """
    
    # 1. Prepare Prediction Metadata
    label = analysis_data.get('label', 'Unknown')
    confidence = analysis_data.get('confidence', 0.0)
    
    probs_list = []
    for cls, prob in analysis_data.get('probabilities', {}).items():
        if prob > 0.01:
            probs_list.append(f"{cls}: {prob:.1%}")
    prob_str = ", ".join(probs_list)

    # 2. Derive Lateralization/Hemisphere
    loc = analysis_data.get('tumor_location')
    hemisphere = "undetermined"
    if loc:
        hemisphere = "Right hemisphere" if loc.get('x', 0.5) > 0.5 else "Left hemisphere"

    # 3. Contextualize Patient Data
    patient_context = f"PATIENT CONTEXT:\n{patient_meta}\n" if patient_meta else ""

    # 4. Construct Prompt
    prompt = f"""<s>[INST] You are a board-certified neuroradiologist and clinical risk expert. 
Generate a structured AI-assisted MRI report based on the following diagnostic data.

MODEL PREDICTION:
- Primary Diagnosis: {label}
- System Confidence: {confidence:.1%}
- Probability Distribution: {prob_str}
- Estimated Lateralization: {hemisphere}

EXTRACTED RISK METRICS (Quantitative Morphology):
- Irregularity/Entropy (0-1): {risk_metrics.get('irregularity_ratio', 0)}
- Activation Area Area: {risk_metrics.get('activation_area', 0):.1%} of brain volume
- Composite NEXUS Risk Score: {risk_metrics.get('risk_score', 0)}/100

{patient_context}

Generate exactly segments for:
1. RADIOLOGICAL IMPRESSION: Professional summary of the AI finding and its morphological implications.
2. RISK ASSESSMENT: Analysis of the irregularity and risk score in clinical context.
3. FOLLOW-UP RECOMMENDATION: Immediate clinical or surgical next steps.

Do NOT provide a definitive medical diagnosis. Frame as AI findings requiring professional correlation. Do not use markdown formatting. [/INST]
"""

    narrative = _generate_text(prompt, max_new_tokens=512, temperature=0.3)
    
    # Heuristic fallback if API response is broken
    if "Error:" in narrative and "API limits" in narrative:
        return (
            f"RADIOLOGICAL IMPRESSION: The AI system identifies {label} with {confidence:.1%} confidence. "
            f"Target area localized to {hemisphere}. Risk Score: {risk_metrics.get('risk_score')}/100."
        )
        
    return narrative
