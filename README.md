# DiNoT: Directional Noise Transport for Semantic Alignment in Diffusion Models

DiNoT (Directional Noise Transport) is a framework for achieving superior semantic alignment in text-to-image diffusion models. By utilizing localized VQA feedback and iterative noise optimization, DiNoT ensures that generated images faithfully represent the complex dependencies and entities described in natural language prompts.

---

## 🛠 Installation & Setup

This repository uses [uv](https://github.com/astral-sh/uv) for extremely fast, reliable dependency management.

### 1. Install `uv`

If you don't have it yet:

```bash
curl -LsSf https://astral-sh/uv/install.sh | sh
```

### 2. Project Setup

Clone the repository and synchronize the environment. `uv` will automatically create a virtual environment and install all necessary dependencies.

```bash
# Clone the repogit@github.com:dino-eccv2026/DiNoT.git
cd DiNoT

# Create environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

---

## 📝 Workflow Step 1: Data Generation

Before generating images, natural language prompts must be decomposed into structured dependency graphs using `data_gen.py`.

1. Add your prompts to a `prompts.txt` file (one per line).
2. Set your OpenAI API key: `export OPENAI_API_KEY='your-key-here'`.
3. Run the generator:

```bash
python data_gen.py --input prompts.txt --output data.json

```

---

## 🎨 Workflow Step 2: Image Generation

Run the optimized pipelines using the structured JSON data. We provide specialized scripts for different base models.

### Stable Diffusion (v1.5 / v2.1)

```bash
python run_dinot_sd.py --json data/drawbench_data.json --output results_sd --sd_model v1.5 --vqa_model llava-v1.5-13b
```

### Stable Diffusion 3.5 Medium

```bash
python run_dinot_sd3.py --json data/drawbench_data.json --output results_sd3 --epochs 2
```

### Stable Diffusion XL 1.0

```bash
python run_dinot_sdxl.py --json data/drawbench_data.json --output results_sdxl --epochs 2
```

---

## 🚀 Benchmarking

To evaluate DiNoT on standard benchmarks, use the pre-processed data in the `data/` directory.

### 📊 DrawBench

```bash
python run_dinot_sdxl.py --json data/drawbench_data.json --output results_drawbench
```

### 📊 GenEval

```bash
python run_dinot_sdxl.py --json data/geneval_data.json --output results_geneval
```

### 📊 PartiPrompts

```bash
python run_dinot_sdxl.py --json data/partiprompt_data.json --output results_partiprompts
```

---

## ⚙️ Configuration Reference
## ⚙️ Configuration Reference

| Argument           | Description                                     | Default                |
| ------------------ | ----------------------------------------------- | ---------------------- |
| `--json`, `-j`     | **Required**. Path to the input JSON file.      | -                      |
| `--output`, `-o`   | Directory to save images and logs.              | `results`              |
| `--num_per_prompt` | Number of images to generate per prompt.        | `1`                    |
| `--epochs`, `-e`   | Optimization iterations per image.              | `2` (SDXL) / `2` (SD3) |
| `--vqa_model`      | VQA model for scoring (e.g., `llava-v1.5-13b`). | `clip-flant5-xl`       |
| `--sd_device`      | CUDA device for the diffusion model.            | `cuda:0`               |
| `--vqa_device`     | CUDA device for the VQA model.                  | `cuda:1`               |
| `--no_ldino`       | Disable optimization (baseline).                | `False`                |

---

## 📂 Project Structure

- `lvqa_dinot/`: Core logic for segmentation, optimization, and scoring.
- `pipelines/`: Custom DiNoT-integrated diffusion pipelines.
- `t2v_metrics/`: Internal library for VQA score calculations.
- `data/`: Benchmark datasets (DrawBench, GenEval, PartiPrompts).
- `ldino_debug/`: Created during runtime; contains segmentation masks and visual feedback.

---

> **Tip:** Check the `initial/` folder within the output directory to verify if Grounded-SAM is correctly identifying your prompt entities.
