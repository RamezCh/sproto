# S-Proto Project

This repository contains the source code for **S-Proto**, a sparse and interpretable prototypical neural network for clinical diagnosis prediction.  
The purpose of this repository is to document the **model architecture, training-to-deployment workflow, and Hugging Face integration**, rather than end-user inference details.

All model usage, inference examples, limitations, and citation information are documented in **`hf/README.md`**.

## Repository Structure

```
.
├── hf/
├── resources/
├── scripts/
```

- **`hf/`**  
  Hugging Face compatible model artifacts, including:
  - `config.json`
  - `model.safetensors`
  - custom configuration and modeling code
  - model card and documentation

- **`resources/`**  
  Local-only resources used during development, such as:
  - original PyTorch Lightning checkpoints (`*.ckpt`)
  - datasets or intermediate artifacts

  These files are intentionally **not tracked by Git** and **not uploaded**.

- **`scripts/`**  
  Utility scripts for:
  - converting Lightning checkpoints to Hugging Face format
  - uploading model artifacts to the Hugging Face Hub
  - validating plug-and-play loading

## Hugging Face Integration Design

S-Proto follows the standard Hugging Face `transformers` extension pattern using custom configuration and modeling code.

### Core Components

**`configuration_sproto.py`**  
Defines `SprotoConfig`, which inherits from `PretrainedConfig`.

This class:
- declares all hyperparameters required to construct the model
- controls architectural choices such as prototype count, dimensions, and sparsity
- is serialized automatically as `config.json`

**`modeling_sproto.py`**  
Defines `SprotoModel`, which inherits from `PreTrainedModel`.

This file:
- builds the neural network modules
- implements the forward pass
- connects the configuration to concrete layers

**`PreTrainedModel` and `PretrainedConfig`**  
Provided by Hugging Face and responsible for:
- standardized save and load behavior
- device and dtype handling
- compatibility with `AutoModel`

### Model Loading Behavior

When calling:

```python
AutoModel.from_pretrained("datexis/sproto", trust_remote_code=True)
```

Hugging Face performs the following steps:

1. Resolves the model location, preferring a local path if present, otherwise falling back to the Hugging Face Hub
2. Loads `config.json` and instantiates `SprotoConfig`
3. Constructs `SprotoModel` using the configuration
4. Loads weights from `model.safetensors`

The `trust_remote_code=True` flag is required because the model uses custom architecture code that is part of this repository.

No PyTorch Lightning checkpoint is required at inference time.

## Development and Deployment Workflow

1. **Training**  
   The model is trained using PyTorch Lightning.  
   The resulting checkpoint is stored locally in `resources/`.

2. **Conversion**  
   The Lightning checkpoint is converted into Hugging Face format:
   - weights are exported to `safetensors`
   - configuration is serialized to `config.json`

3. **Upload**  
   The contents of `hf/` are uploaded to the Hugging Face Hub.

4. **Inference**  
   Users load the model directly from the Hub using standard Hugging Face APIs.

## Setup

### Dependencies

This project uses Poetry for dependency management.

```bash
poetry install
```

### Environment Variables

Create a `.env` file in the repository root containing your Hugging Face token:

```
HF_TOKEN=your_hugging_face_token
```

## Scripts

### Convert Lightning Checkpoint

`scripts/convert_to_safetensor.py` converts a local Lightning checkpoint into Hugging Face compatible artifacts.

```bash
poetry run python scripts/convert_to_safetensor.py
```

This script:
- loads a `.ckpt` file from `resources/`
- exports `model.safetensors`
- generates `config.json`
- writes outputs to `hf/`

### Upload to Hugging Face Hub

`scripts/upload_to_hf.py` uploads the contents of `hf/` to a Hugging Face repository.

```bash
poetry run python scripts/upload_to_hf.py --repo <username>/<repo-name> [--dry-run]
```

- `--repo` specifies the target repository
- `--dry-run` performs a validation without uploading files

### Plug-and-Play Load Test

`scripts/plug_play_test.py` verifies that the model can be loaded via `from_pretrained`.

```bash
poetry run python scripts/plug_play_test.py
```

The script first attempts to load from the local `hf/` directory and falls back to the Hugging Face Hub if not found.

## Notes

- PyTorch Lightning checkpoints remain local and are never distributed
- All public distribution uses `safetensors` for safety and integrity
- End-user inference, usage examples, limitations, and citations are documented in [**hf/README.md**](./hf/README.md)

## Model Documentation

For full model details, interpretability notes, limitations, and citation information, see:

[**hf/README.md**](./hf/README.md)

![S-Proto Overview](hf/overview.png)