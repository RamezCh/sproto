from transformers import AutoConfig, AutoTokenizer, AutoModel

# 1. Load Config
config = AutoConfig.from_pretrained("RamezCh/sproto", trust_remote_code=True)

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("RamezCh/sproto")

# 3. Load Model
model = AutoModel.from_pretrained("RamezCh/sproto", trust_remote_code=True)

# 4. Run Inference
encoding = tokenizer(
    ["Test sentence.", "Another example."],
    padding=True,
    truncation=True,
    return_tensors="pt"
)

outputs = model(**encoding)
print(outputs)
