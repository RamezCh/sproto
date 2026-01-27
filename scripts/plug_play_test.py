from transformers import AutoTokenizer, AutoModel
import torch

# 1. Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModel.from_pretrained("DATEXIS/sproto", trust_remote_code=True)
model.eval()

# 2. Sample Data
text_input = [
    "CHIEF COMPLAINT: Right Carotid Artery Stenosis. PRESENT ILLNESS: Ms. ___ is a ___ year old woman with hyperlipidemia, cirrhosis with esophageal varices, alcoholism, COPD, left eye blindness, and right carotid stenosis status post right carotid endarterectomy."
]

# 3. Tokenize standardly
inputs = tokenizer(
    text_input,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# 4. Prepare the "tokens" argument
tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in inputs["input_ids"]]

# 5. Run Inference
with torch.no_grad():
    output = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs.get("token_type_ids"),
        tokens=tokens
    )

# 6. Access results from the returned dictionary
logits = output["logits"]
max_indices = output["max_indices"]
metadata = output["metadata"]

print("Inference successful!")
print(f"Logits shape: {logits.shape}")
print(f"Logits: {logits}")
print(f"Max indices: {max_indices}")
print(f"Meta-Data: {metadata}")