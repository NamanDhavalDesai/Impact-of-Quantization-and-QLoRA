# The Impact of Quantization on Swedish Language Understanding: Evaluation & Mitigation via QLoRA

![Llama-3](https://img.shields.io/badge/Model-Llama--3--8B--Instruct-blue)
![NLP](https://img.shields.io/badge/Task-NLI%20%2F%20WiC-green)
![Quantization](https://img.shields.io/badge/Optimization-4--bit%20NF4-orange)
![QLoRA](https://img.shields.io/badge/Fine--Tuning-QLoRA-red)

## 📌 Project Overview
As Large Language Models (LLMs) scale, 4-bit quantization has become the standard for efficient deployment. However, its impact on non-English languages—specifically morphologically rich languages like Swedish—is often overlooked. 

This project evaluates **Llama-3-8B** across 16-bit, 8-bit and 4-bit precisions using Swedish (SuperLim) and English (GLUE) benchmarks. We identify a specific failure mode called **"Neutral Class Collapse"** and demonstrate how **QLoRA fine-tuning** on a translated instruction dataset can restore lost linguistic nuance.

## 🚀 Key Discovery: "Neutral Class Collapse"
Our research found that aggressive 4-bit quantization (NF4) does not degrade all capabilities equally. Instead, it specifically destroys the model's ability to handle **ambiguity**:
*   In Natural Language Inference (NLI) tasks, the model's F1 score for the "Neutral" class dropped from **0.23 to 0.02**.
*   The model effectively regressed into a binary classifier (Entailment vs. Contradiction), losing the "nuance bits" required for non-binary logic.

## 🛠️ Methodology

### 1. Model Configurations
*   **M1 (16-bit)**: Baseline BFloat16.
*   **M2 (8-bit)**: Quantized via `LLM.int8()`.
*   **M3 (4-bit)**: Quantized via `NF4` with Double Quantization.
*   **M4 (4-bit + QLoRA)**: 4-bit base model fine-tuned on Swedish instruction data.

### 2. Fine-Tuning (QLoRA)
We mitigated the 4-bit degradation by fine-tuning on the **ALPACA-SWEDISH-REFINED** dataset.
*   **Adapters**: Query, Key, Value and Output projection layers.
*   **Parameters**: Rank ($r$) = 16, $\alpha$ = 32.
*   **Optimization**: Paged AdamW with gradient checkpointing.

### 3. Constrained Evaluation Pipeline
To ensure robust metrics, we used the **Outlines library** for regex-guided generation. This forced the quantized models to adhere to valid label tokens, eliminating parsing errors caused by quantization-induced "instruction-following decay."

## 📊 Results Summary

| Model | Swedish NLI (F1) | Swedish WiC (F1) | Key Observation |
| :--- | :---: | :---: | :--- |
| **16-bit (M1)** | 0.623 | 0.608 | Baseline Performance |
| **4-bit (M3)** | 0.545 | 0.642 | **12% Drop** in NLI; Neutral class collapse |
| **4-bit Adapter (M4)** | **0.631** | 0.632 | **Full recovery** of Swedish nuance |

**The "Language Tax":** While QLoRA restored Swedish performance, we observed **catastrophic forgetting** in English lexical tasks (English WiC dropped from 0.64 to 0.53), suggesting a trade-off between language-specific adaptation and general knowledge preservation.

## 📂 Repository Structure
*   `code/`: Contains the full implementation pipeline (see internal `README.md` for execution details).
*   `Report.pdf`: The full technical study.
*   `Presentation.pptx`: Summary of the entire technical study.

## 🚀 Getting Started
Refer to the `README.md` inside the `code/` directory for instructions on setting up the environment, running the quantization scripts and reproducing the QLoRA training.

## 👥 Contributors
*   **Naman Dhaval Desai**
*   **Noah Marquez Vara**
*   **Yingjue Ma**
