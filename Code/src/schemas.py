from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any


class ModelConfig(BaseModel):
    name: str = Field(..., description="Name of the model configuration")
    model_id: str = Field(..., description="HuggingFace model ID")
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    compute_dtype: str = "bfloat16"
    quant_type: str = "nf4"
    use_double_quant: bool = True
    adapter_path: Optional[str] = None


class TaskConfig(BaseModel):
    name: str
    dataset_id: str
    subset: Optional[str] = None
    split: str = "validation"
    text_cols: List[str]
    label_col: str
    label_mapping: Dict[int, str]
    eval_label_order: Optional[List[str]] = None
    output_type: str
    num_labels: int = 3
    prompt_template: Optional[str] = None


class ExperimentConfig(BaseModel):
    model: ModelConfig
    task: TaskConfig
    output_dir: str = "results"
    seed: int = 42
    sample_limit: Optional[int] = None  # None means use full dataset
    few_shot_k: int = 3
