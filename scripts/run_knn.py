import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import torch
from recbole.quick_start import run_recbole

from plot_knn import plot_knn_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

# Prevent RecBole from interpreting this script's arguments.
sys.argv = sys.argv[:1]
torch.set_num_threads(4)

run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")
mode = "smoke" if args.smoke else "knn"
output_dir = Path("results") / mode / run_id
figure_dir = Path("figures") / mode / run_id
output_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

# Preserve the settings and code used in this run.
shutil.copy2("configs/common.yaml", output_dir / "common.yaml")
shutil.copy2(__file__, output_dir / "run_knn.py")
shutil.copy2(Path(__file__).with_name("plot_knn.py"), output_dir / "plot_knn.py")

k_values = [100] if args.smoke else [1, 5, 10, 100, 500]
rows = []

for method, model_label in [("item", "ItemKNN"), ("user", "UserKNN")]:
    for k in k_values:
        experiment = f"{model_label}_k{k}"
        print(f"\n===== START {experiment} =====", flush=True)

        # Give each experiment a fresh logger.
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

        overrides = {
            "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision"],
            "knn_method": method,
            "k": k,
            "checkpoint_dir": str(Path("saved") / run_id / experiment),
        }

        (output_dir / f"{experiment}_settings.json").write_text(
            json.dumps(overrides, indent=2),
            encoding="utf-8",
        )

        result = run_recbole(
            model="ItemKNN",
            dataset="ml-100k",
            config_file_list=["configs/common.yaml"],
            config_dict=overrides,
            saved=True,
        )

        row = {
            "run_id": run_id,
            "model": model_label,
            "knn_method": method,
            "k": k,
            "seed": 2020,
        }
        for split, key in [
            ("valid", "best_valid_result"),
            ("test", "test_result"),
        ]:
            for metric, value in result[key].items():
                row[f"{split}_{metric}"] = float(value)

        rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
        print(json.dumps(row, indent=2), flush=True)
        print(f"===== DONE {experiment} =====", flush=True)

df = pd.DataFrame(rows)

if not args.smoke:
    plot_knn_metrics(df, figure_dir)

    # Choose each model's k using validation performance.
    best = df.loc[df.groupby("model")["valid_ndcg@10"].idxmax()]
    best.to_csv(output_dir / "best_by_validation.csv", index=False)

    print("\nBest settings selected by validation NDCG@10:")
    print(best[[
        "model", "k", "valid_ndcg@10",
        "test_ndcg@10", "test_recall@10", "test_precision@10",
        "test_mrr@10", "test_hit@10"
    ]].to_string(index=False))

print(f"\nResults: {output_dir}")
print(f"Figures: {figure_dir}")
print("ALL REQUESTED RUNS COMPLETED", flush=True)
