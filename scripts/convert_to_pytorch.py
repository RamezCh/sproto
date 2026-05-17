import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from hf.configuration_sproto import SprotoConfig


CHECKPOINT_PATH = Path(__file__).parent.parent / "resources" / "ckpt-epoch=573.ckpt"
HF_DIR = Path(__file__).parent.parent / "hf"


def load_lightning_checkpoint(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    return state_dict, checkpoint.get("hyper_parameters", {})


def map_state_dict_keys(state_dict):
    mapped_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith("bert."):
            mapped_key = f"module.bert.{key[5:]}"
        elif key == "prototype_vectors":
            mapped_key = "module.prototype_vectors"
        elif key == "attention_vectors":
            mapped_key = "module.attention_vectors"
        elif key.startswith("linear."):
            mapped_key = f"module.linear.{key[7:]}"
        elif key == "final_linear":
            mapped_key = "module.final_linear"
        elif key == "num_prototypes_per_class":
            mapped_key = "module.num_prototypes_per_class"
        else:
            continue
        
        mapped_state_dict[mapped_key] = value
    
    return mapped_state_dict


def extract_config(hyper_parameters):
    import torch
    
    def to_python(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.tolist()
        return value
    
    config_params = {
        "pretrained_model": hyper_parameters.get("pretrained_model", "bert-base-uncased"),
        "num_classes": to_python(hyper_parameters.get("num_classes", 2)),
        "label_order_path": hyper_parameters.get("label_order_path"),
        "use_attention": hyper_parameters.get("use_attention", True),
        "use_global_attention": hyper_parameters.get("use_global_attention", False),
        "dot_product": hyper_parameters.get("dot_product", False),
        "normalize": hyper_parameters.get("normalize"),
        "final_layer": hyper_parameters.get("final_layer", False),
        "reduce_hidden_size": hyper_parameters.get("reduce_hidden_size"),
        "num_prototypes_per_class": to_python(hyper_parameters.get("num_prototypes_per_class", 1)),
    }
    
    return config_params


def convert_checkpoint(checkpoint_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    state_dict, hyper_parameters = load_lightning_checkpoint(checkpoint_path)
    
    mapped_state_dict = map_state_dict_keys(state_dict)
    
    print(f"Mapped {len(mapped_state_dict)} keys")
    
    pytorch_path = output_dir / "pytorch_model.bin"
    torch.save(mapped_state_dict, str(pytorch_path))
    print(f"Saved weights to {pytorch_path}")
    
    config_params = extract_config(hyper_parameters)
    config = SprotoConfig(**config_params)
    
    config_path = output_dir / "config.json"
    config.to_json_file(str(config_path))
    print(f"Saved config to {config_path}")
    
    print("\nConfiguration:")
    for key, value in config_params.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Convert Lightning checkpoint to HuggingFace format")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH),
        help="Path to Lightning checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(HF_DIR),
        help="Output directory for HuggingFace artifacts"
    )
    
    args = parser.parse_args()
    
    convert_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()