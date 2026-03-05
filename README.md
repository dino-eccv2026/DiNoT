# L-DINO-CoT: Localized Directional Noise Optimization with Chain-of-Thought

This repository contains the implementation of **L-DINO-CoT**, a method for improving semantic alignment in text-to-image diffusion models through localized VQA scoring and directional noise optimization.

## 🚀 Key Features

- **Localized VQA Optimization**: Uses Grounded-SAM to segment entities and optimize their alignment individually.
- **Hierarchical Scoring**: Evaluates prompts based on a dependency graph of entities, attributes, and relations.
- **Multi-Model Support**: Works across Stable Diffusion v1.x, v2.x, SD3.5, and SDXL 1.0.
- **Flexible VQA Integration**: Compatible with multiple state-of-the-art VQA models.

---

## 📂 Directory Structure

- `pipelines/`: Specialized DiNO pipelines with optimization loops.
  - `pipeline_sd_dino.py`: For SD v1.4, v1.5, v2.0, v2.1, v2.2.
  - `pipeline_sd3_dino.py`: For Stable Diffusion 3.5 Medium.
  - `pipeline_sdxl_dino.py`: For Stable Diffusion XL 1.0.
- `lvqa_dino/`: Core logic for L-DINO-CoT (segmentation, optimization, scoring).
- `t2v_metrics/`: Local library for VQA score computation.
- `run_dino_sd.py`: Runner for SD v1.x/v2.x models.
- `run_dino_sd3.py`: Runner for SD 3.5.
- `run_dino_sdxl.py`: Runner for SDXL 1.0.
- `data_gen.py`: GPT-based tool for decomposing prompts into dependency graphs.
- `sample_data/`: Contains `single_prompt.json` for immediate testing.

---

## 🛠 Supported Models

### Diffusion Models

| Model Type           | Identifier                                 | Version               |
| -------------------- | ------------------------------------------ | --------------------- |
| Stable Diffusion     | `v1.4`, `v1.5`, `v2.0`, `v2.1`, `v2.2`     | Standard SD checkouts |
| Stable Diffusion 3.5 | `stabilityai/stable-diffusion-3.5-medium`  | SD3.5 Medium          |
| Stable Diffusion XL  | `stabilityai/stable-diffusion-xl-base-1.0` | SDXL 1.0 Base         |

### VQA Models (for scoring)

- `clip-flant5-xl` (Default, recommended for speed/memory)
- `llava-v1.5-13b` (Higher accuracy)
- `sharegpt4v-13b`
- `instructblip-flant5-xl`

---

## 📝 Data Generation (`data_gen.py`)

This script uses OpenAI's GPT-4o-mini to decompose natural language prompts into the structured **dependency graph** required by the optimization pipeline.

### Preparation

1. Create a `prompts.txt` file with one prompt per line.
2. Ensure `OPENAI_API_KEY` is set in your environment.

### Execution

```bash
python data_gen.py --input prompts.txt --output data.json
```

The resulting `data.json` will contain everything needed for the runner scripts.

---

## 🎨 Image Generation

All runner scripts require a JSON file (standardize via `data_gen.py`) and an output directory.

### 1. Stable Diffusion (v1.x / v2.x)

```bash
python run_dino_sd.py \
    --json sample_data/single_prompt.json \
    --output results_sd \
    --sd_model v1.5 \
    --vqa_model llava-v1.5-13b
```

### 2. Stable Diffusion 3.5

```bash
python run_dino_sd3.py \
    --json sample_data/single_prompt.json \
    --output results_sd3 \
    --epochs 20
```

### 3. Stable Diffusion XL 1.0

```bash
python run_dino_sdxl.py \
    --json sample_data/single_prompt.json \
    --output results_sdxl \
    --epochs 2
```

---

## ⚙️ Configuration & Arguments

| Argument                 | Description                                | Default                     |
| ------------------------ | ------------------------------------------ | --------------------------- |
| `--json`, `-j`           | **Required**. Path to the input JSON file. | -                           |
| `--output`, `-o`         | Directory to save images and logs.         | `results`                   |
| `--num_per_prompt`, `-n` | Images to generate per prompt.             | `4`                         |
| `--epochs`, `-e`         | Optimization iterations per image.         | `2` (SDXL/SD) or `20` (SD3) |
| `--vqa_model`            | VQA model for scoring/optimization.        | `clip-flant5-xl`            |
| `--sd_device`            | CUDA device for diffusion.                 | `cuda:0`                    |
| `--vqa_device`           | CUDA device for VQA.                       | `cuda:1`                    |
| `--no_ldino`             | Disable optimization and run baseline.     | `False`                     |

---

## 🔍 Debugging

When `--save_debug` is active (default in scripts), each prompt directory will contain an `ldino_debug/` folder with:

- `initial/`: Segmentation masks generated at the start.
- `epoch_N/`: Visual feedback of segmentation per optimization epoch.
