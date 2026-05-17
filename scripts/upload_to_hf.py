import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo


HF_DIR = Path(__file__).parent.parent / "hf"


def upload_to_hub(repo_id, hf_dir, dry_run=False):
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    
    if not token:
        raise ValueError("HF_TOKEN not found in .env file")
    
    api = HfApi(token=token)
    root_dir = hf_dir.parent  # project root (one level above hf/)
    sproto_dir = root_dir / "sproto"
    
    print(f"Repository: {repo_id}")
    print(f"Uploading HF artifacts from: {hf_dir}")
    print(f"Uploading sproto package from: {sproto_dir}")
    
    if dry_run:
        print("\n--- DRY RUN ---")
        print("Files that would be uploaded from hf/:")
        for file_path in hf_dir.rglob("*"):
            if file_path.is_file() and "__pycache__" not in str(file_path):
                print(f"  {file_path.relative_to(hf_dir)}")
        print("Files that would be uploaded from sproto/ (as sproto/):")
        for file_path in sproto_dir.rglob("*"):
            if file_path.is_file() and "__pycache__" not in str(file_path):
                print(f"  sproto/{file_path.relative_to(sproto_dir)}")
        print("--- END DRY RUN ---\n")
        return
    
    create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
    print(f"Created/verified repository: {repo_id}")
    
    # Upload hf/ folder (model weights, config, tokenizer, wrapper code)
    api.upload_folder(
        folder_path=str(hf_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["__pycache__/**", "*.pyc"],
    )
    
    # Upload sproto/ package so users don't need a separate install.
    # HF caches trust_remote_code files under:
    #   ~/.cache/huggingface/modules/transformers_modules/<repo_id>/
    # Uploading sproto/ to repo root places it next to modeling_sproto.py
    # in that cache directory, so `from sproto.model.multi_proto import ...` resolves.
    api.upload_folder(
        folder_path=str(sproto_dir),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="sproto",
        ignore_patterns=["__pycache__/**", "*.pyc"],
    )
    
    print(f"\nSuccessfully uploaded to https://huggingface.co/{repo_id}")



def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Target repository (e.g., username/repo-name)"
    )
    parser.add_argument(
        "--hf-dir",
        type=str,
        default=str(HF_DIR),
        help="Directory containing HuggingFace artifacts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform validation without uploading"
    )
    
    args = parser.parse_args()
    
    upload_to_hub(args.repo, Path(args.hf_dir), args.dry_run)


if __name__ == "__main__":
    main()