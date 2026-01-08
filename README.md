# S-Proto Project

This repository contains code for **S-Proto**, a sparse and interpretable prototypical network for clinical diagnosis prediction.

## Structure

- **`hf/`**: Contains the Hugging Face model files (model weights, config, model card).
  - Note: `overview.png` is located here.
- **`resources/`**: Contains resource files like checkpoints and datasets.
- **`scripts/`**: Utility scripts for converting models and uploading to Hugging Face.

## Setup

1.  **Install dependencies**:
    This project uses [Poetry](https://python-poetry.org/).
    ```bash
    poetry install
    ```

2.  **Environment Variables**:
    Create a `.env` file in the root directory (or use `.env` example if provided) with your Hugging Face token:
    ```
    HF_TOKEN=your_hugging_face_token
    ```

## Scripts

### 1. Converting Lightning Checkpoint to Safetensors

The script `scripts/convert_to_safetensor.py` converts a PyTorch Lightning checkpoint (located in `resources/`) to the Hugging Face `safetensors` format and saves it to `hf/`.

**Usage**:
```bash
poetry run python scripts/convert_to_safetensor.py
```
This will:
- Load the checkpoint from `resources/`.
- Convert the weights.
- Save `model.safetensors` and config files to `hf/` with the correct metadata.

### 2. Uploading to Hugging Face

The script `scripts/upload_to_hf.py` uploads the contents of the `hf/` directory to a Hugging Face Hub repository.

**Usage**:
```bash
poetry run python scripts/upload_to_hf.py --repo <your-username>/<repo-name> [--dry-run]
```
- `--repo`: The target repository ID (e.g., `datexis/sproto`).
- `--dry-run`: Use this flag to simulate the upload without actually transferring files.

### 3. Testing Inference

The script `scripts/plug_play_test.py` demonstrates how to load the model for inference. It will try to load from the local `hf/` directory first, and fallback to the Hugging Face Hub if not found.

**Usage**:
```bash
poetry run python scripts/plug_play_test.py
```

## Model Details

For detailed model documentation, limitations, and citation, please refer to the Model Card in [hf/README.md](hf/README.md).

![S-Proto Overview](hf/overview.png)
