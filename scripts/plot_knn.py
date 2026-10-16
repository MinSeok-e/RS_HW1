"""Draw annotated KNN plots from saved metrics without rerunning experiments."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("ndcg@10", "NDCG@10", (-9, 9), "right", "bottom"),
    ("recall@10", "Recall@10", (9, -1), "left", "center"),
    ("precision@10", "Precision@10", (0, -12), "center", "top"),
    ("mrr@10", "MRR@10", (0, 9), "center", "bottom"),
    ("hit@10", "Hit@10", (0, 9), "center", "bottom"),
]


def plot_knn_metrics(data, figure_dir):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Use the same vertical scale for both algorithms.
    columns = [f"test_{metric}" for metric, *_ in METRICS]
    upper = min(1.05, float(data[columns].max().max()) + 0.1)

    for model_label in ["ItemKNN", "UserKNN"]:
        subset = data[data["model"] == model_label].sort_values("k")
        positions = list(range(len(subset)))
        fig, ax = plt.subplots(figsize=(9.5, 6))

        for metric, label, offset, horizontal, vertical in METRICS:
            values = subset[f"test_{metric}"]
            (line,) = ax.plot(positions, values, marker="o", label=label)
            for x, value in zip(positions, values):
                ax.annotate(
                    f"{value:.4f}",
                    (x, value),
                    xytext=offset,
                    textcoords="offset points",
                    ha=horizontal,
                    va=vertical,
                    fontsize=10,
                    color=line.get_color(),
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
                )

        ax.set_xticks(positions, subset["k"].astype(str))
        ax.set_xlim(-0.45, len(subset) - 0.55)
        ax.set_ylim(0, upper)
        ax.set_title(f"{model_label} on MovieLens 100K", pad=43)
        ax.set_xlabel("Number of neighbors (k)")
        ax.set_ylabel("Test metric value")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False)
        fig.tight_layout()
        destination = figure_dir / f"{model_label}.png"
        fig.savefig(destination, dpi=200)
        plt.close(fig)
        print(f"Saved: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_knn_metrics(pd.read_csv(args.results_dir / "metrics.csv"), args.figure_dir)
