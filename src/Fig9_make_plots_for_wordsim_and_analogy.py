import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():

    ps = [1, 2, 5, 10, 20, 50, 100, 200, 300]

    input_dir = Path("output/evaluation")
    csv_names = [
        "wordsim.csv",
        "ica_MSR_analogy.csv",
        "ica_Google_analogy.csv",
        "pca_MSR_analogy.csv",
        "pca_Google_analogy.csv",
    ]

    df = pd.DataFrame()
    for csv_name in csv_names:
        csv_path = input_dir / csv_name
        df_ = pd.read_csv(csv_path)
        df = pd.concat([df, df_], axis=0)

    results = {}
    task_types = ["similarity", "analogy"]
    for ti, task_type in enumerate(task_types):
        task_df = df[df["task_type"] == task_type]
        emb2scores = {}
        emb_types = task_df["emb_type"].unique()
        logger.info(f"emb_types: {emb_types}, task_type: {task_type}")
        for emb_type in emb_types:
            emb_df = task_df[task_df["emb_type"] == emb_type]
            scores = []
            for p in ps:
                p_df = emb_df[emb_df["p"] == p]
                # score
                if task_type == "similarity":
                    score = np.mean(p_df["spearman"].values)
                elif task_type == "analogy":
                    score = np.mean(p_df["top1-acc"].values)
                scores.append(score)
            emb2scores[emb_type] = scores

        results[task_type] = emb2scores

    # plot
    fig, axes = plt.subplots(1, len(task_types), figsize=(15, 7))

    ts = 35
    fs = 30
    ls = 25
    legend_s = 30
    for i, (task_type, emb2scores) in enumerate(results.items()):
        ax = axes[i]
        for emb_type in ["ica", "pca"]:
            if emb_type in emb2scores:
                scores = emb2scores[emb_type]
            else:
                continue

            if emb_type == "pca":
                linestyle = "--"
                color = "blue"
                label = "PCA"
                ax.plot(
                    ps,
                    scores,
                    label=label,
                    marker="o",
                    linewidth=3,
                    markersize=10,
                    linestyle=linestyle,
                    color=color,
                )
            elif emb_type == "ica":
                linestyle = "-"
                color = "orange"
                label = "ICA"
                ax.plot(
                    ps,
                    scores,
                    label=label,
                    marker="o",
                    linewidth=3,
                    markersize=10,
                    linestyle=linestyle,
                    color=color,
                    zorder=10,
                )

        if task_type == "similarity":
            ax.set_title("Word Similarity", fontsize=ts, pad=15)
            ax.set_xlabel("Number of Non-Zero Axes", fontsize=fs)
            ax.set_ylabel(r"Spearman's $\rho$", fontsize=fs)
        elif task_type == "analogy":
            ax.set_title("Analogy", fontsize=ts, pad=15)
            ax.set_xlabel("Number of Non-Zero Axes", fontsize=fs)
            ax.set_ylabel("Top1 acc.", fontsize=fs)

        # x is log scale
        ax.set_xscale("log")
        if i == 0:
            ax.legend(loc="lower right", fontsize=legend_s)

        # tick
        ax.tick_params(labelsize=ls, which="major", length=10)
        ax.tick_params(axis="x", which="minor", length=5)

        # ylim
        ax.set_ylim(-0.02, 0.65)

    fig.subplots_adjust(
        left=0.09, right=0.99, bottom=0.08, top=0.9, wspace=0.25, hspace=0.1
    )
    output_dir = Path("output/camera_ready_images")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "wordsim_and_analogy_plots.pdf"
    logger.info(f"save to {output_path}")
    plt.savefig(output_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
