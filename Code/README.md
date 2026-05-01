# Low-Bit Quantization & QLoRA Study: English vs Swedish Llama-3

This project evaluates the impact of quantization (FP16 → INT8 → INT4) on Llama-3's cross-lingual performance in English and Swedish, and attempts to recover lost performance using QLoRA fine-tuning.

## Project Structure

```
├── src/                    # Core modules
│   ├── model_manager.py    # Model loading with quantization
│   ├── data_loader.py      # Dataset loading and preprocessing
│   ├── evaluator.py        # Evaluation with Outlines structured generation
│   └── schemas.py          # Pydantic config schemas
├── scripts/
│   ├── train_qlora.py      # QLoRA fine-tuning script
│   └── aggregate_results.py # Results aggregation
├── conf/                   # Hydra configuration
│   ├── config.yaml         # Main config
│   ├── model/              # Model configs (16bit, 8bit, 4bit, 4bit_adapter)
│   └── task/               # Task configs (nli_en, nli_sv, wic_en, wic_sv)
└── results/                # Output directory
```

## Tasks

| Task | English | Swedish | Type |
|------|---------|---------|------|
| **NLI** | SuperGLUE CB | sbx/superlim-2 swenli | 3-class |
| **WiC** | SuperGLUE WiC | sbx/superlim-2 swewic | Binary |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Login to Hugging Face (for Llama-3 access)
huggingface-cli login
```

## Complete Workflow

### Step 1: Benchmark Baseline & Quantized Models (M1, M2, M3)

Evaluate all models on all tasks:

```bash
python3 run_eval.py -m model=16bit,8bit,4bit task=nli_en,nli_sv,wic_en,wic_sv
```

This runs **12 experiments** (3 models × 4 tasks) and saves results to `results/`.

### Step 2: Swedish QLoRA Fine-Tuning (Create M4)

Fine-tune the 4-bit model on Swedish data to recover cross-lingual performance:

```bash
python3 scripts/train_qlora.py
```

This saves LoRA adapters to `results/adapters/m4_swedish/`.

### Step 3: Evaluate Fine-Tuned Model (M4)

Run evaluation with the fine-tuned adapter on Swedish tasks:

```bash
python3 run_eval.py -m model=4bit_adapter task=nli_sv,wic_sv
```

(Optional) Also evaluate M4 on English to check for catastrophic forgetting:

```bash
python3 run_eval.py -m model=4bit_adapter task=nli_en,wic_en
```

### Step 4: Aggregate & Compare Results

Generate a summary table comparing all models:

```bash
python3 scripts/aggregate_results.py
```

This outputs:
- **Global Performance Table**: Macro F1 scores for all model/task combinations
- **Degradation Analysis**: Performance drop from 16-bit to 4-bit

## Model Configurations

| Config | Description | Precision |
|--------|-------------|-----------|
| `16bit` | FP16/BF16 baseline (M1) | Full precision |
| `8bit` | INT8 quantized (M2) | 8-bit |
| `4bit` | INT4-NF4 quantized (M3) | 4-bit |
| `4bit_adapter` | INT4 + Swedish QLoRA (M4) | 4-bit + LoRA |

## Configuration Options

Edit `conf/config.yaml` to adjust:

```yaml
sample_limit: 50    # Number of samples per task (increase for final run)
few_shot_k: 3       # Number of few-shot examples
seed: 42            # Random seed
```
Edit `train_qlora.py` to adjust:

```
def train_qlora(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_id="neph1/Alpaca-Lora-GPT4-Swedish-Refined",
    output_dir="results/adapters/m4_swedish",
    epochs=1,
    max_samples=2000,  # Limit samples for faster training (~30 mins)
):
```