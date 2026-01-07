import argparse
from huggingface_hub import HfApi, upload_folder

def upload_model(repo_name: str, token: str, dry_run: bool = False):
    """Upload model files to a Hugging Face repository.

    Args:
        repo_name: Full repository name, e.g. "datexis/sproto".
        token: Hugging Face access token with write permissions.
        dry_run: If True, only prints the files that would be uploaded.
    """
    api = HfApi()
    # Ensure repo exists (creates if not)
    if not dry_run:
        api.create_repo(
            repo_id=repo_name,
            token=token,
            private=True,
            repo_type="model",
            exist_ok=True,
        )

    if dry_run:
        print("[DRY RUN] Would upload repository contents")
        return

    upload_folder(
        repo_id=repo_name,
        folder_path=".",
        repo_type="model",
        token=token,
        allow_patterns=[
            "model.safetensors",
            "modeling_sproto.py",
            "configuration_sproto.py",
            "config.json",
            "tokenizer_config.json",
            "vocab.txt",
            "merges.txt",
            "README.md",
            "model_card.md",
        ],
        commit_message="Upload sproto model",
    )

    print("Upload complete.")

def main():
    parser = argparse.ArgumentParser(description="Upload sproto model to Hugging Face Hub")
    parser.add_argument("--repo", required=True, help="Target HF repository (e.g., datexis/sproto)")
    parser.add_argument("--token", default="YOUR_HF_TOKEN", help="HF access token (placeholder if not set)")
    parser.add_argument("--dry-run", action="store_true", help="Show files to be uploaded without performing upload")
    args = parser.parse_args()
    upload_model(repo_name=args.repo, token=args.token, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
