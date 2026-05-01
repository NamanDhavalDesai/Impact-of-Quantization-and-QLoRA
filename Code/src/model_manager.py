import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import logging
import gc
from src.schemas import ModelConfig

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_config = None

    def load_model(self, config: ModelConfig):
        """
        Loads the model based on the provided ModelConfig Pydantic object.
        """
        # If we already have this model config loaded, skip
        if self.current_config == config and self.current_model is not None:
            return self.current_model, self.current_tokenizer

        # Unload previous
        self.unload_model()

        logger.info(f"Loading model: {config.name} ({config.model_id})")

        self.current_tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        if self.current_tokenizer.pad_token is None:
            self.current_tokenizer.pad_token = self.current_tokenizer.eos_token

        device_map = "auto"
        quant_config = None

        # Determine quantization
        if config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, config.compute_dtype),
                bnb_4bit_quant_type=config.quant_type,
                bnb_4bit_use_double_quant=config.use_double_quant,
            )
        elif config.load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load Base Model
        if quant_config:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                quantization_config=quant_config,
                torch_dtype=getattr(torch, config.compute_dtype),
                device_map=device_map,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=getattr(torch, config.compute_dtype),
                device_map=device_map,
            )

        # Load Adapter if present (M4)
        if config.adapter_path:
            logger.info(f"Loading adapters from {config.adapter_path}")
            model = PeftModel.from_pretrained(model, config.adapter_path)

        if self.current_tokenizer.pad_token_id is not None:
            model.config.pad_token_id = self.current_tokenizer.pad_token_id
            # Also update generation_config if it exists
            if (
                hasattr(model, "generation_config")
                and model.generation_config is not None
            ):
                model.generation_config.pad_token_id = (
                    self.current_tokenizer.pad_token_id
                )

        self.current_model = model
        self.current_config = config

        return self.current_model, self.current_tokenizer

    def unload_model(self):
        if self.current_model:
            logger.info("Unloading model...")
            # Move to CPU before deleting to help allocator
            try:
                self.current_model.to("cpu")
            except:
                pass
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_config = None

            # Aggressive cleanup
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
