import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__) 

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator that masks everything before the response template.
    Vendored here to ensure stability on SLURM/L4 environments.
    """
    def __init__(
        self,
        response_template: Union[str, List[int]],
        tokenizer: Any,
        mlm: bool = False,
        ignore_index: int = -100,
        pad_to_multiple_of: Optional[int] = None,
    ):
        super().__init__(tokenizer, mlm=mlm, mlm_probability=0, pad_to_multiple_of=pad_to_multiple_of)
        self.response_template = response_template
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer
        
        # Pre-encode response template
        if isinstance(response_template, str):
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            self.response_token_ids = response_template

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # Iterate over batch to mask user prompts
        for i in range(len(batch["labels"])):
            response_token_ids_start_idx = None
            
            # Find the start of the response template in the sequence
            # We iterate backwards or forwards? Llama-3 template is unique, usually appears once.
            for idx in range(len(batch["labels"][i]) - len(self.response_token_ids) + 1):
                if np.array_equal(
                    batch["labels"][i][idx : idx + len(self.response_token_ids)].cpu().numpy(), 
                    self.response_token_ids
                ):
                    response_token_ids_start_idx = idx
                    break

            if response_token_ids_start_idx is None:
                # If template not found, warn and mask everything (safety)
                # This happens if truncation cuts off the answer
                batch["labels"][i, :] = self.ignore_index
            else:
                # Mask everything BEFORE the response start
                # We include the template itself in the loss? Usually NO, we mask up to the end of template.
                response_start = response_token_ids_start_idx + len(self.response_token_ids)
                batch["labels"][i, :response_start] = self.ignore_index

        return batch

def train_qlora(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    dataset_id="neph1/Alpaca-Lora-GPT4-Swedish-Refined",
    output_dir="results/adapters/m4_swedish",
    epochs=1,
    max_samples=2000,  # Limit samples for faster training (~30 mins)
):
    logger.info("Starting QLoRA Fine-tuning...")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for SFTTrainer

    # 2. Load Model in 4-bit (M3 state)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map="auto"
    )

    # Prepare for training
    model.config.use_cache = False  # Disable cache during training
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ],
    )

    # 4. Load Dataset (with limit for faster training)
    dataset = load_dataset(dataset_id, split="train")

    if max_samples and len(dataset) > max_samples:
        logger.info(f"Limiting dataset from {len(dataset)} to {max_samples} samples")
        dataset = dataset.shuffle(seed=42).select(range(max_samples))

    # Preprocess dataset - create "text" column for SFTTrainer
    def format_sample(sample):
        instruction = sample["instruction"]
        input_text = sample.get("input", "")
        response = sample["output"]

        user_content = instruction
        if input_text:
            user_content += f"\nInput: {input_text}"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]

        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    # Calculate expected training time
    effective_batch_size = 4 * 4  # batch_size * gradient_accumulation
    total_steps = (len(dataset) * epochs) // effective_batch_size
    logger.info(
        f"Training: {len(dataset)} samples, {epochs} epoch(s), ~{total_steps} steps"
    )

    # Define the pattern that marks the start of the assistant's answer
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, 
        tokenizer=tokenizer
    )

    # 5. Sft Config
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        logging_steps=10,
        learning_rate=2e-4,
        bf16=True,
        max_grad_norm=0.3,
        num_train_epochs=epochs,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_length=512,
        packing=False,
    )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=collator,
        processing_class=tokenizer,
        args=sft_config,
    )

    trainer.train()

    logger.info(f"Saving adapters to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return output_dir


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )

    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for Swedish")
    parser.add_argument(
        "--model_id",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model ID",
    )
    parser.add_argument(
        "--dataset_id",
        default="neph1/Alpaca-Lora-GPT4-Swedish-Refined",
        help="Swedish instruction dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="results/adapters/m4_swedish",
        help="Output directory for adapters",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=3000,
        help="Max training samples (default 3000 for ~30 min training)",
    )

    args = parser.parse_args()

    train_qlora(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        epochs=args.epochs,
        max_samples=args.max_samples,
    )
