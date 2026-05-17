import argparse
import torch
from pathlib import Path

from transformers import AutoModel, AutoConfig


HF_DIR = Path(__file__).parent.parent / "hf"


def test_local_load(hf_dir):
    print(f"Testing local load from: {hf_dir}")
    
    config = AutoConfig.from_pretrained(str(hf_dir), trust_remote_code=True)
    print(f"  Config loaded: num_classes={config.num_classes}")
    
    model = AutoModel.from_pretrained(str(hf_dir), trust_remote_code=True)
    print(f"  Model loaded successfully")
    
    return model, config


def test_hub_load(repo_id):
    print(f"Testing hub load from: {repo_id}")
    
    config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
    print(f"  Config loaded: num_classes={config.num_classes}")
    
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    print(f"  Model loaded successfully")
    
    return model, config


def test_forward(model, config):
    batch_size = 2
    seq_len = 10
    num_classes = config.num_classes
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    
    print(f"  Forward pass successful")
    print(f"    logits shape: {outputs['logits'].shape}")
    print(f"    expected shape: ({batch_size}, {num_classes})")
    
    assert outputs["logits"].shape == (batch_size, num_classes), "Shape mismatch!"
    print("  Shape validation passed")


def main():
    parser = argparse.ArgumentParser(description="Test plug-and-play model loading")
    parser.add_argument(
        "--repo",
        type=str,
        default="RamezCh/sproto",
        help="HuggingFace repository to test"
    )
    parser.add_argument(
        "--hf-dir",
        type=str,
        default=str(HF_DIR),
        help="Local HuggingFace directory"
    )
    
    args = parser.parse_args()
    
    hf_dir = Path(args.hf_dir)
    
    if hf_dir.exists() and (hf_dir / "config.json").exists():
        print("=" * 50)
        print("Testing LOCAL load")
        print("=" * 50)
        model, config = test_local_load(hf_dir)
        test_forward(model, config)
        print("\nLocal load test PASSED!")
    else:
        print(f"Local hf directory not found at {hf_dir}, skipping local test")
    
    print("\n" + "=" * 50)
    print("Testing HUB load")
    print("=" * 50)
    
    try:
        model, config = test_hub_load(args.repo)
        test_forward(model, config)
        print("\nHub load test PASSED!")
    except Exception as e:
        print(f"\nHub load test FAILED: {e}")
        print("This is expected if the model hasn't been uploaded yet")


if __name__ == "__main__":
    main()