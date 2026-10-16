import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from bpr_regularized import RegularizedBPR


ROOT = Path(__file__).resolve().parents[1]

# (embedding_size, reg_lambda): shared condition (64, 0.1) runs once.
CONDITIONS = [
    (32, 0.1),
    (64, 0.1),
    (128, 0.1),
    (256, 0.1),
    (64, 0.01),
    (64, 0.5),
]


def plot_ndcg(data, x_column, xlabel, title, destination):
    data = data.sort_values(x_column)
    x = data[x_column].tolist()
    y = data["test_ndcg@10"].tolist()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:g}" for value in x])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test NDCG@10")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.margins(y=0.2)
    ax.set_ylim(bottom=0)

    for x_value, y_value in zip(x, y):
        ax.annotate(
            f"{y_value:.4f}",
            (x_value, y_value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    torch.set_num_threads(4)

    run_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_dir = ROOT / "results" / "bpr" / run_id
    figure_dir = ROOT / "figures" / "bpr" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_dir.mkdir(parents=True, exist_ok=False)

    # Preserve the exact code and settings used for this experiment.
    for relative_path in [
        "configs/common.yaml",
        "configs/bpr.yaml",
        "scripts/bpr_regularized.py",
        "scripts/run_bpr.py",
    ]:
        source = ROOT / relative_path
        shutil.copy2(source, output_dir / source.name)

    rows = []

    for index, (embedding_size, reg_lambda) in enumerate(CONDITIONS, 1):
        experiment = f"BPR_dim{embedding_size}_lambda{reg_lambda:g}"
        print(
            f"\n===== START {index}/{len(CONDITIONS)}: {experiment} =====",
            flush=True,
        )
        started = perf_counter()

        # Give each condition its own RecBole log file.
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

        config = Config(
            model="BPR",
            dataset="ml-100k",
            config_file_list=[
                str(ROOT / "configs/common.yaml"),
                str(ROOT / "configs/bpr.yaml"),
            ],
            config_dict={
                "embedding_size": embedding_size,
                "reg_lambda": reg_lambda,
                "require_pow": True,
                "model_implementation": "RegularizedBPR",
                "checkpoint_dir": str(
                    ROOT / "saved" / "bpr" / run_id / experiment
                ),
            },
        )

        # Save the fully resolved configuration, including library defaults.
        (output_dir / f"{experiment}_config.json").write_text(
            json.dumps(config.final_config_dict, indent=2, default=str),
            encoding="utf-8",
        )

        # Reset before dataset splitting to use the same split in every run.
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)
        logger.info("Experiment: %s", experiment)
        logger.info(config)

        dataset = create_dataset(config)
        logger.info(dataset)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # Initialize the tested custom model using RecBole's BPR defaults.
        init_seed(config["seed"], config["reproducibility"])
        model = RegularizedBPR(config, train_data._dataset).to(
            config["device"]
        )
        logger.info(model)

        trainer = Trainer(config, model)
        try:
            _, best_valid_result = trainer.fit(
                train_data,
                valid_data,
                saved=True,
                show_progress=False,
            )
            # Test the checkpoint selected by validation NDCG@10.
            test_result = trainer.evaluate(
                test_data,
                load_best_model=True,
                show_progress=False,
            )
        finally:
            trainer.tensorboard.close()

        row = {
            "run_id": run_id,
            "experiment": experiment,
            "model": "RegularizedBPR",
            "dataset": "ml-100k",
            "embedding_size": embedding_size,
            "reg_lambda": reg_lambda,
            "seed": config["seed"],
            "max_epochs": config["epochs"],
            "epochs_trained": len(trainer.train_loss_dict),
            "elapsed_seconds": round(perf_counter() - started, 2),
            "checkpoint": trainer.saved_model_file,
        }
        for split, metrics in [
            ("valid", best_valid_result),
            ("test", test_result),
        ]:
            for metric, value in metrics.items():
                row[f"{split}_{metric}"] = float(value)

        rows.append(row)

        # Save after each completed condition.
        pd.DataFrame(rows).to_csv(
            output_dir / "metrics.csv", index=False
        )
        logger.info("Best validation: %s", best_valid_result)
        logger.info("Test result: %s", test_result)
        print(json.dumps(row, indent=2), flush=True)
        print(
            f"===== DONE {index}/{len(CONDITIONS)}: {experiment} =====",
            flush=True,
        )

    results = pd.DataFrame(rows)

    embedding_results = results[
        results["reg_lambda"] == 0.1
    ].sort_values("embedding_size")
    regularization_results = results[
        results["embedding_size"] == 64
    ].sort_values("reg_lambda")

    embedding_results.to_csv(
        output_dir / "embedding_sweep.csv", index=False
    )
    regularization_results.to_csv(
        output_dir / "regularization_sweep.csv", index=False
    )

    plot_ndcg(
        embedding_results,
        "embedding_size",
        "Embedding size",
        "BPR on MovieLens 100K (reg_lambda = 0.1)",
        figure_dir / "BPR_embedding_size.png",
    )
    plot_ndcg(
        regularization_results,
        "reg_lambda",
        "Regularization coefficient",
        "BPR on MovieLens 100K (embedding_size = 64)",
        figure_dir / "BPR_reg_lambda.png",
    )

    # Select among the six tested conditions using validation only.
    best = results.loc[[results["valid_ndcg@10"].idxmax()]]
    best.to_csv(output_dir / "best_by_validation.csv", index=False)

    columns = [
        "embedding_size", "reg_lambda", "epochs_trained",
        "valid_ndcg@10", "test_ndcg@10",
        "test_recall@10", "test_precision@10",
    ]
    print("\nAll six conditions:", flush=True)
    print(results[columns].to_string(index=False), flush=True)
    print("\nBest condition selected by validation NDCG@10:", flush=True)
    print(best[columns].to_string(index=False), flush=True)
    print(f"\nResults: {output_dir}", flush=True)
    print(f"Figures: {figure_dir}", flush=True)
    print("BPR FULL EXPERIMENTS COMPLETED (6/6)", flush=True)


if __name__ == "__main__":
    main()
