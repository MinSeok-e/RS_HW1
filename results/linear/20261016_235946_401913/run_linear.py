import json
import logging
import shutil
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import yaml
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_logger, init_seed


ROOT = Path(__file__).resolve().parents[1]
KNN_RUN = ROOT / "results" / "knn" / "20261016_012128"
METRICS = ["ndcg@10", "recall@10", "precision@10"]


def build_model(config, dataset, logger):
    model_name = config["model"]
    started = perf_counter()
    finished = threading.Event()

    def report_elapsed():
        while not finished.wait(15):
            logger.info(
                "%s model fitting is running: %.0f seconds elapsed",
                model_name,
                perf_counter() - started,
            )

    logger.info("%s: starting model fitting", model_name)
    reporter = threading.Thread(target=report_elapsed, daemon=True)
    reporter.start()
    try:
        model = get_model(model_name)(config, dataset).to(config["device"])
    finally:
        finished.set()
        reporter.join()

    logger.info(
        "%s: model fitting completed in %.2f seconds",
        model_name,
        perf_counter() - started,
    )
    return model


def main():
    torch.set_num_threads(4)

    # Ensure the comparison uses the same common settings as the KNN run.
    current_common = yaml.safe_load(
        (ROOT / "configs/common.yaml").read_text(encoding="utf-8")
    )
    knn_common = yaml.safe_load(
        (KNN_RUN / "common.yaml").read_text(encoding="utf-8")
    )
    if current_common != knn_common:
        raise ValueError(
            "Common settings differ from the saved KNN experiment. "
            "Check configs/common.yaml before comparing results."
        )

    knn_best = pd.read_csv(KNN_RUN / "best_by_validation.csv")
    selected = knn_best[knn_best["model"] == "ItemKNN"]
    if len(selected) != 1:
        raise ValueError("Expected exactly one validation-selected ItemKNN row.")
    baseline = selected.iloc[0]

    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_dir = ROOT / "results" / "linear" / run_id
    figure_dir = ROOT / "figures" / "linear" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)

    # Preserve the code, configurations, and source of the KNN comparison.
    for relative_path in [
        "configs/common.yaml",
        "configs/slim.yaml",
        "configs/ease.yaml",
        "scripts/run_linear.py",
    ]:
        source = ROOT / relative_path
        shutil.copy2(source, output_dir / source.name)

    shutil.copy2(
        KNN_RUN / "best_by_validation.csv",
        output_dir / "knn_best_by_validation.csv",
    )
    shutil.copy2(
        KNN_RUN / "common.yaml",
        output_dir / "knn_common.yaml",
    )

    baseline_row = {
        "run_id": str(baseline["run_id"]),
        "model": "ItemKNN",
        "parameters": f"k={int(baseline['k'])}, knn_method=item",
        "seed": int(baseline["seed"]),
    }
    for split in ["valid", "test"]:
        for metric in METRICS:
            key = f"{split}_{metric}"
            baseline_row[key] = float(baseline[key])

    rows = [baseline_row]
    models = [
        ("SLIMElastic", "configs/slim.yaml"),
        ("EASE", "configs/ease.yaml"),
    ]

    for index, (model_name, config_path) in enumerate(models, 1):
        print(f"\n===== START {index}/2: {model_name} =====", flush=True)

        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        config = Config(
            model=model_name,
            dataset="ml-100k",
            config_file_list=[
                str(ROOT / "configs/common.yaml"),
                str(ROOT / config_path),
            ],
            config_dict={
                # These models learn their item weights during construction.
                "epochs": 1,
                "checkpoint_dir": str(
                    ROOT / "saved" / "linear" / run_id / model_name
                ),
            },
        )

        (output_dir / f"{model_name}_config.json").write_text(
            json.dumps(config.final_config_dict, indent=2, default=str),
            encoding="utf-8",
        )

        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)
        logger.info(config)

        dataset = create_dataset(config)
        logger.info(dataset)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        init_seed(config["seed"], config["reproducibility"])
        model = build_model(config, train_data._dataset, logger)
        logger.info(model)

        trainer_class = get_trainer(config["MODEL_TYPE"], model_name)
        trainer = trainer_class(config, model)

        try:
            _, best_valid_result = trainer.fit(
                train_data,
                valid_data,
                saved=True,
                show_progress=False,
            )
            test_result = trainer.evaluate(
                test_data,
                load_best_model=True,
                show_progress=False,
            )
        finally:
            trainer.tensorboard.close()

        model_settings = yaml.safe_load(
            (ROOT / config_path).read_text(encoding="utf-8")
        )
        row = {
            "run_id": run_id,
            "model": model_name,
            "parameters": json.dumps(model_settings, sort_keys=True),
            "seed": config["seed"],
        }
        for split, metrics in [
            ("valid", best_valid_result),
            ("test", test_result),
        ]:
            for metric, value in metrics.items():
                row[f"{split}_{metric}"] = float(value)

        rows.append(row)

        # Save immediately after each completed model.
        pd.DataFrame(rows[1:]).to_csv(
            output_dir / "metrics.csv", index=False
        )
        pd.DataFrame(rows).to_csv(
            output_dir / "comparison.csv", index=False
        )

        logger.info("Validation: %s", best_valid_result)
        logger.info("Test result: %s", test_result)
        print(json.dumps(row, indent=2), flush=True)
        print(f"===== DONE {index}/2: {model_name} =====", flush=True)

    comparison = pd.DataFrame(rows)
    labels = [
        f"ItemKNN\n(k={int(baseline['k'])})",
        "SLIM",
        "EASE",
    ]

    fig, axes = plt.subplots(figsize=(12, 4), ncols=3)
    for ax, metric in zip(axes, METRICS):
        values = comparison[f"test_{metric}"].to_numpy()
        bars = ax.bar(
            labels,
            values,
            color=["#4C78A8", "#F58518", "#54A24B"],
        )
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_title(metric.upper())
        ax.set_ylabel("Test score")
        ax.set_ylim(0, max(float(values.max()) * 1.2, 0.01))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.suptitle("MovieLens 100K: ItemKNN vs SLIM vs EASE")
    fig.tight_layout()
    fig.savefig(
        figure_dir / "ItemKNN_SLIM_EASE.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    columns = [
        "model", "valid_ndcg@10",
        "test_ndcg@10", "test_recall@10", "test_precision@10",
    ]
    print("\nModel comparison:", flush=True)
    print(comparison[columns].to_string(index=False), flush=True)
    print(f"\nResults: {output_dir}", flush=True)
    print(f"Figures: {figure_dir}", flush=True)
    print("LINEAR EXPERIMENTS COMPLETED (2/2)", flush=True)


if __name__ == "__main__":
    main()
