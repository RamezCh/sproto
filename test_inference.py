import sys
import torch
from transformers import AutoTokenizer
from hf.configuration_sproto import SprotoConfig
from hf.modeling_sproto import SprotoModel

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    print("Loading model config...")
    config = SprotoConfig.from_pretrained("hf")

    print("Instantiating model...")
    model = SprotoModel(config)

    print("Loading model weights...")
    state_dict = torch.load("hf/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    text = """CHIEF COMPLAINT: depression, chest pain and vomiting

PRESENT ILLNESS: The patient is a 53-year-old woman with history of hypertension, diabetes, and depression. Unfortunately her husband left her 10 days prior to admission and she developed severe anxiety and depression. She was having chest pains along with significant vomiting and diarrhea. Of note, she had a nuclear stress test performed in February of this year, which was normal.

PHYSICAL EXAMINATION: Significant for her being afebrile. Apparently there was one temperature registered mildly high at 100. Her blood pressure was 140/82, heart rate 83, oxygen saturation was 100%. She was tearful. HEART: Heart sounds were regular. LUNGS: Clear. ABDOMEN: Soft. Apparently there were some level of restlessness and acathexia. She was also pacing."""

    print("\nProcessing input text...")
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # The Sproto model requires the raw token strings for its clinical section masking logic
    tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokens=tokens
        )

    logits = outputs.logits
    # Apply sigmoid to get probabilities (since loss is BCE)
    probs = torch.sigmoid(logits)[0]

    print("\n--- Inference Results ---")
    
    # Load labels and mappings from s-proto-demo
    try:
        import json
        with open("../s-proto-demo/data/labels.txt", "r") as f:
            labels = f.read().strip().split("\n")
        with open("../s-proto-demo/data/icd_10_mappings.json", "r") as f:
            icd_mapping = json.load(f)
        with open("../s-proto-demo/data/thresholds_per_label.json", "r") as f:
            threshold_mapping = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load label mapping files: {e}")
        labels = None
        threshold_mapping = None

    # Filter predictions based on specific thresholds per label (fallback to 0.20 if missing or exactly 0.0)
    if labels and threshold_mapping:
        threshold_tensor = torch.zeros(len(labels))
        for idx, label in enumerate(labels):
            val = threshold_mapping.get(label, 0.20)
            # A threshold of exactly 0.0 is invalid because sigmoid always outputs > 0.0
            if val == 0.0:
                val = 0.20
            threshold_tensor[idx] = val
        
        predicted_indices = torch.where(probs > threshold_tensor)[0]
        print("\nAll Predicted Diagnoses (Probability > Label-Specific Threshold):")
    else:
        THRESHOLD = 0.20
        predicted_indices = torch.where(probs > THRESHOLD)[0]
        print(f"\nAll Predicted Diagnoses (Probability > {THRESHOLD}):")
    if len(predicted_indices) == 0:
        print("No diagnoses predicted above the threshold.")
    else:
        results = []
        for idx in predicted_indices:
            idx_val = idx.item()
            prob = probs[idx_val].item()
            
            if labels and idx_val < len(labels):
                icd_code = labels[idx_val]
                description = icd_mapping.get(icd_code, "Unknown Description")
                results.append((icd_code, description, prob, idx_val))
            else:
                results.append((f"Class_{idx_val:04d}", "Unknown Description", prob, idx_val))
                
        # Sort alphabetically by ICD-10 code (the first element of the tuple)
        results.sort(key=lambda x: x[0])
        
        for icd_code, description, prob, idx_val in results:
            if icd_code.startswith("Class_"):
                print(f"- Class Index {idx_val}: {prob:.4f} probability")
            else:
                print(f"- {icd_code} ({description}): {prob:.4f} probability")

if __name__ == "__main__":
    main()
