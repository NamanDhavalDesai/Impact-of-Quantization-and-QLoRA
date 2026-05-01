import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import logging
import os
import json
from src.schemas import ExperimentConfig, ModelConfig, TaskConfig
from src.model_manager import ModelManager
from src.data_loader import DatasetLoader
from src.evaluator import Evaluator

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Validate Config with Pydantic
    try:
        # Convert OmegaConf to dict, then validate
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        # We need to restructure slightly because Hydra flattens or nests differently
        # based on composition. Our schema expects {model: ..., task: ...}
        # which matches the config structure.

        # Ensure label_mapping keys are ints (YAML might parse as strings)
        if "label_mapping" in cfg_dict["task"]:
            cfg_dict["task"]["label_mapping"] = {
                int(k): v for k, v in cfg_dict["task"]["label_mapping"].items()
            }

        experiment_config = ExperimentConfig(**cfg_dict)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise e

    # 2. Log Configuration
    logger.info(
        f"Starting Run: Model={experiment_config.model.name}, Task={experiment_config.task.name}"
    )

    # 3. Load Resources
    model_manager = ModelManager()

    try:
        model, tokenizer = model_manager.load_model(experiment_config.model)

        loader = DatasetLoader(experiment_config.task.name, cfg_dict["task"])
        dataset = loader.get_samples(limit=experiment_config.sample_limit)
        few_shot = loader.get_few_shot_examples(k=experiment_config.few_shot_k)

        if not dataset:
            logger.warning("Dataset empty, skipping.")
            return

        # 4. Run Evaluation
        evaluator = Evaluator(model, tokenizer, experiment_config.task)
        results = evaluator.evaluate(dataset, few_shot_examples=few_shot)

        # 5. Log results immediately (visible in SLURM output)
        logger.info("=" * 60)
        logger.info(f"RESULTS: {experiment_config.model.name} / {experiment_config.task.name}")
        logger.info(f"  Samples evaluated: {len(results.get('predictions', []))}")
        logger.info(f"  Macro F1: {results['macro_f1']:.4f}")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  Labels: {results['labels_order']}")
        logger.info("  Confusion Matrix:")
        for i, row in enumerate(results['confusion_matrix']):
            logger.info(f"    {results['labels_order'][i]}: {row}")
        logger.info("=" * 60)

        # 6. Add metadata for post-training analysis
        results['metadata'] = {
            'model_name': experiment_config.model.name,
            'task_name': experiment_config.task.name,
            'sample_limit': experiment_config.sample_limit,
            'few_shot_k': experiment_config.few_shot_k,
            'total_samples': len(results.get('predictions', [])),
        }

        # 7. Save Results to Hydra's output directory
        hydra_cfg = HydraConfig.get()
        output_dir = hydra_cfg.runtime.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir, f"results_{experiment_config.task.name}.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Run completed. Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Run failed: {e}")
        raise e
    finally:
        model_manager.unload_model()


if __name__ == "__main__":
    main()
