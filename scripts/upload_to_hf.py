import os
import argparse
from huggingface_hub import HfApi, upload_folder
from dotenv import load_dotenv

# load .env from root directory (one folder up)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

def upload_model(repo_name: str, token: str, dry_run: bool = False):
    api = HfApi()
    if not dry_run:
        api.create_repo(
            repo_id=repo_name,
            token=token,
            private=True,
            repo_type="model",
            exist_ok=True,
        )
    allow_patterns = [
        "modeling_sproto.py",
        "configuration_sproto.py",
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt"
        "overview.png",
        "LICENSE",
        "README.md",
    ]

    if dry_run:
        print("[DRY RUN] The following files would be uploaded:")
        from fnmatch import fnmatch
        folder_path = os.path.join(os.path.dirname(__file__), "..", "hf")
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), folder_path)
                if any(fnmatch(rel_path, pattern) for pattern in allow_patterns):
                    print(f" - {rel_path}")
        return

    upload_folder(
        repo_id=repo_name,
        folder_path=os.path.join(os.path.dirname(__file__), "..", "hf"),
        repo_type="model",
        token=token,
        allow_patterns=allow_patterns,
        commit_message="Update ReadME",
    )

def main():
    print("Uploading model to Hugging Face")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError("HF_TOKEN environment variable not set")

    upload_model(repo_name=args.repo, token=token, dry_run=args.dry_run)

    print("Model uploaded successfully")

if __name__ == "__main__":
    main()