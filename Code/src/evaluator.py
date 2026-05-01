import torch
import logging
from tqdm import tqdm
import outlines
from outlines import models
from outlines import Generator as OutlinesGenerator
from outlines.types import regex
from src.schemas import TaskConfig
import pandas as pd
import re

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model, tokenizer, task_config: TaskConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = task_config
        self.labels_map = task_config.label_mapping  # {0: "Entailment", ...}
        # Invert for checking
        self.valid_labels = list(self.labels_map.values())

        # Initialize Outlines model wrapper
        self.outlines_model = models.Transformers(model, tokenizer)

    def format_input(self, sample):
        """Constructs the input text from sample columns."""
        text_cols = self.config.text_cols
        parts = []
        for col in text_cols:
            val = sample.get(col, "")
            parts.append(f"{col.capitalize()}: {val}")
        return "\n".join(parts)

    def format_label(self, sample):
        """Returns the string representation of the label."""
        label_idx = sample[self.config.label_col]
        return self.labels_map.get(label_idx, str(label_idx))

    def build_prompt(self, sample, few_shot_examples):
        """
        Builds the full prompt string using Llama-3 formatting.
        Outlines works best with a single string prompt for generation.
        """
        # Use task-specific prompt template if available
        task_instruction = self.config.prompt_template or f"Task: {self.config.name}."

        system_message = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a linguistic expert. Answer with only the label.\n"
            f"{task_instruction}\n"
            f"Valid labels: {', '.join(self.valid_labels)}.<|eot_id|>"
        )

        prompt = system_message

        for ex in few_shot_examples:
            user_text = self.format_input(ex)
            assistant_text = self.format_label(ex)
            prompt += (
                f"\n<|start_header_id|>user<|end_header_id|>\n{user_text}<|eot_id|>"
                f"\n<|start_header_id|>assistant<|end_header_id|>\n{assistant_text}<|eot_id|>"
            )

        # Target sample
        target_text = self.format_input(sample)
        prompt += (
            f"\n<|start_header_id|>user<|end_header_id|>\n{target_text}<|eot_id|>"
            f"\n<|start_header_id|>assistant<|end_header_id|>\n"
        )

        return prompt

    def evaluate(self, dataset, few_shot_examples=[]):
        predictions = []
        references = []

        logger.info(
            f"Starting Outlines structured evaluation on {len(dataset)} samples..."
        )

        # Create generator
        # Build regex: (Label1|Label2|Label3)
        options_regex = (
            f"({'|'.join([re.escape(label) for label in self.valid_labels])})"
        )

        # Use Outlines Generator directly with regex constraint
        generator = OutlinesGenerator(self.outlines_model, regex(options_regex))

        prompts = [self.build_prompt(sample, few_shot_examples) for sample in dataset]

        preds = []
        logger.info("Running sequential generation...")
        for p in tqdm(prompts):
            try:
                preds.append(generator(p))
            except Exception as e:
                logger.error(f"Generation failed for prompt: {e}")
                preds.append("Error")

        # Filter out errors
        valid_indices = [i for i, p in enumerate(preds) if p != "Error"]
        preds = [preds[i] for i in valid_indices]
        dataset = [dataset[i] for i in valid_indices]  # Align dataset to preds

        predictions = preds

        for i, sample in enumerate(dataset):
            ref_val = sample[self.config.label_col]  # int
            ref_str = self.labels_map.get(ref_val, str(ref_val))
            references.append(ref_str)

        # Determine label order for metrics
        if self.config.eval_label_order:
            labels_list = self.config.eval_label_order
        else:
            # Fallback to sorted values
            labels_list = sorted(list(set(self.labels_map.values())))

        # Metrics
        from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
        import numpy as np

        # Ensure all labels in predictions/references are present in labels_list
        unique_refs = set(references)
        unique_preds = set(predictions)
        known_labels = set(labels_list)

        unknown_refs = unique_refs - known_labels
        if unknown_refs:
            # Convert set to sorted list for W&B JSON serialization
            unknown_list = sorted(list(unknown_refs))
            logger.error(
                f"Found labels in references that are not in config label list: {unknown_list}"
            )
            labels_list.extend(unknown_list)

        unknown_preds = unique_preds - known_labels
        if unknown_preds:
            unknown_list = sorted(list(unknown_preds))
            logger.error(
                f"Found labels in predictions that are not in config label list: {unknown_list}"
            )
            labels_list.extend(unknown_list)

        # Use string labels for everything
        macro_f1 = f1_score(
            references,
            predictions,
            labels=labels_list,
            average="macro",
            zero_division=0,
        )
        acc = accuracy_score(references, predictions)
        cm = confusion_matrix(references, predictions, labels=labels_list)
        
        # Per-class F1 scores for detailed analysis
        per_class_f1 = f1_score(
            references,
            predictions,
            labels=labels_list,
            average=None,
            zero_division=0,
        )
        per_class_f1_dict = {label: float(score) for label, score in zip(labels_list, per_class_f1)}

        results = {
            "macro_f1": macro_f1,
            "accuracy": acc,
            "per_class_f1": per_class_f1_dict,
            "confusion_matrix": cm.tolist(),
            "labels_order": labels_list,  # Save order for visualization
            "predictions": predictions,
            "references": references,
            "num_samples": len(predictions),
        }

        return results
