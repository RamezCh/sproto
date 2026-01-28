from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    use_fast=True
)

tokenizer.save_pretrained("./tokenizer")
