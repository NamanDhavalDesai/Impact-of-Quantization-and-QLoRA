import os
import json
import glob
from collections import defaultdict
from tabulate import tabulate
import pandas as pd


def aggregate_results(base_dir="results"):
    """
    Crawls the Hydra output directory to find result JSONs and aggregates them.
    Assumes Hydra structure: results/YYYY-MM-DD/HH-MM-SS/results_{task}.json
    OR multirun structure: results/YYYY-MM-DD/HH-MM-SS/X/results_{task}.json
    """

    # Search for all result files
    # Recursive search might be needed if using multirun
    # With new config, Hydra creates subdirs like:
    # results/DATE/TIME/results_*.json (if single run)
    # results/DATE/TIME/0/results_*.json (if multirun)

    files = glob.glob(f"{base_dir}/**/results_*.json", recursive=True)

    data = defaultdict(dict)  # {model_name: {task_name: f1_score}}

    print(f"Found {len(files)} result files.")

    for fpath in files:
        try:
            with open(fpath, "r") as f:
                res = json.load(f)

            # We need to identify which model/task this belongs to.
            # The file name is results_{task}.json.
            # The model info is usually in the config.yaml saved by Hydra in .hydra/config.yaml
            # adjacent to the results file.

            dir_path = os.path.dirname(fpath)
            config_path = os.path.join(dir_path, ".hydra", "config.yaml")

            if not os.path.exists(config_path):
                # Maybe flat structure?
                continue

            import yaml

            with open(config_path, "r") as cf:
                cfg = yaml.safe_load(cf)

            model_name = cfg["model"]["name"]
            task_name = cfg["task"]["name"]

            f1 = res.get("macro_f1", 0)
            data[model_name][task_name] = f1

        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    # Generate Table
    if not data:
        print("No results found.")
        return

    df = pd.DataFrame(data).transpose()  # Rows=Model, Cols=Task

    # Sort models in logical order: 16bit -> 8bit -> 4bit -> 4bit_adapter
    model_order = ["16bit", "8bit", "4bit", "4bit_adapter"]
    existing_models = [m for m in model_order if m in df.index]
    other_models = [m for m in df.index if m not in model_order]
    df = df.reindex(existing_models + other_models)

    # Sort tasks: English first, then Swedish
    task_order = ["nli_en", "nli_sv", "wic_en", "wic_sv"]
    existing_tasks = [t for t in task_order if t in df.columns]
    other_tasks = [t for t in df.columns if t not in task_order]
    df = df[existing_tasks + other_tasks]

    print("\n=== Global Performance (Macro F1) ===")
    print(tabulate(df, headers="keys", tablefmt="grid"))

    # Calculate Deltas if M1 and M3 exist
    if "16bit" in df.index and "4bit" in df.index:
        print("\n=== Degradation Analysis (16bit -> 4bit) ===")
        m1 = df.loc["16bit"]
        m3 = df.loc["4bit"]

        delta = ((m1 - m3) / m1 * 100).fillna(0)
        print(delta.to_frame(name="% Drop").to_markdown())


if __name__ == "__main__":
    aggregate_results()
