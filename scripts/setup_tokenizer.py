from transformers import AutoTokenizer

def setup_tokenizer():
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    save_directory = "."
    
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Saving tokenizer to {save_directory}...")
    tokenizer.save_pretrained(save_directory)
    print("Tokenizer saved successfully.")

if __name__ == "__main__":
    setup_tokenizer()
