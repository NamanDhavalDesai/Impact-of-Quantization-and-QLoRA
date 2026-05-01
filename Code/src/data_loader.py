import logging
from datasets import load_dataset
import random
import json

logger = logging.getLogger(__name__)


def parse_swewic(dataset):
    """
    Parse sbx/superlim-2 swewic dataset where data is embedded as JSON in 'idx' column.
    Extracts: word, sentence1, sentence2, label (same_sense -> 1, different_sense -> 0)
    Uses HuggingFace map/filter for efficiency.
    """

    def _extract_fields(sample):
        """Extract fields from JSON embedded in 'idx' column."""
        idx_val = sample.get("idx")
        if not idx_val or not isinstance(idx_val, str):
            return {"word": "", "sentence1": "", "sentence2": "", "label": -1}
        try:
            item = json.loads(idx_val)
            word = item.get("first", {}).get("word", {}).get("text", "")
            sentence1 = item.get("first", {}).get("context", "")
            sentence2 = item.get("second", {}).get("context", "")
            label_str = item.get("label", "")

            # Map: "same_sense" -> 1 (True), "different_sense" -> 0 (False)
            if label_str == "same_sense":
                label = 1
            elif label_str == "different_sense":
                label = 0
            else:
                label = -1  # Mark invalid for filtering

            return {
                "word": word,
                "sentence1": sentence1,
                "sentence2": sentence2,
                "label": label,
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return {"word": "", "sentence1": "", "sentence2": "", "label": -1}

    # Use map for parallel processing, then filter invalid samples
    parsed = dataset.map(_extract_fields, remove_columns=dataset.column_names)
    parsed = parsed.filter(lambda x: x["label"] != -1)
    return parsed


class DatasetLoader:
    def __init__(self, task_name, task_config):
        self.task_name = task_name
        self.config = task_config
        self.dataset = None
        self.train_dataset = []

        self.load_data()

    def load_data(self):
        """Loads the dataset and prepares few-shot examples."""
        dataset_id = self.config["dataset_id"]
        subset = self.config.get("subset")
        split = self.config.get("split", "validation")

        logger.info(f"Loading task {self.task_name}: {dataset_id}/{subset} [{split}]")

        try:
            # Load dataset
            if subset:
                raw_dataset = load_dataset(dataset_id, subset, split=split)
                try:
                    raw_train = load_dataset(dataset_id, subset, split="train")
                except Exception:
                    raw_train = []
            else:
                raw_dataset = load_dataset(dataset_id, split=split)
                raw_train = load_dataset(dataset_id, split="train")

            # Special handling for swewic (JSON embedded format)
            if dataset_id == "sbx/superlim-2" and subset == "swewic":
                logger.info("Applying swewic JSON format preprocessing...")
                self.dataset = parse_swewic(raw_dataset)
                self.train_dataset = parse_swewic(raw_train) if raw_train else []
                return

            # Standard processing
            self.dataset = raw_dataset
            self.train_dataset = raw_train if raw_train else []

            if len(self.dataset) > 0:
                # Filter out header rows (seen in sbx/superlim-2 swenli)
                first_label = self.dataset[0][self.config["label_col"]]
                if (
                    isinstance(first_label, str)
                    and first_label.lower() == self.config["label_col"]
                ):
                    logger.info("Detected header row in dataset, filtering...")
                    self.dataset = self.dataset.filter(
                        lambda x: x[self.config["label_col"]].lower()
                        != self.config["label_col"]
                    )
                    if self.train_dataset:
                        self.train_dataset = self.train_dataset.filter(
                            lambda x: x[self.config["label_col"]].lower()
                            != self.config["label_col"]
                        )

                # Convert string labels to integers if needed
                if len(self.dataset) > 0:
                    sample_label = self.dataset[0][self.config["label_col"]]
                    if isinstance(sample_label, str):
                        logger.info(
                            "Dataset has string labels. Converting to integers..."
                        )
                        str_to_int = {
                            v.lower(): k
                            for k, v in self.config["label_mapping"].items()
                        }

                        def encode_labels(example):
                            label = example[self.config["label_col"]]
                            if isinstance(label, str):
                                return {
                                    self.config["label_col"]: str_to_int.get(
                                        label.lower(), -1
                                    )
                                }
                            return example

                        self.dataset = self.dataset.map(encode_labels)
                        if self.train_dataset:
                            self.train_dataset = self.train_dataset.map(encode_labels)

        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}/{subset}: {e}")
            self.dataset = []
            self.train_dataset = []
            raise e

    def get_samples(self, limit=None):
        """Returns samples for evaluation."""
        if not self.dataset or len(self.dataset) == 0:
            return []

        ds = self.dataset
        if limit:
            ds = ds.select(range(min(len(ds), limit)))

        return ds

    def get_few_shot_examples(self, k=3, seed=42):
        """Returns k random examples from training set for few-shot prompting."""
        if not self.train_dataset or len(self.train_dataset) == 0:
            return []

        random.seed(seed)
        indices = random.sample(
            range(len(self.train_dataset)), min(k, len(self.train_dataset))
        )
        examples = [self.train_dataset[i] for i in indices]
        return examples
