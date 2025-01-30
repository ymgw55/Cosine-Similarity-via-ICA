import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from utils import pos_direct

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def model_name2title(model_name):
    if model_name == "glove":
        return "GloVe"
    elif model_name == "fasttext":
        return "fastText"
    elif model_name == "bert-base-uncased":
        return "BERT-base"
    elif model_name == "roberta-base":
        return "RoBERTa-base"
    elif model_name == "gpt2":
        return "GPT-2"
    elif model_name == "EleutherAI-pythia-160m":
        return "Pythia-160m"


def main():
    dynamic_models = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "EleutherAI-pythia-160m",
    ]

    model2normed_embed = {}

    input_dir = Path("output/embeddings/")
    for model_name in dynamic_models:
        logger.info(model_name)

        input_path = input_dir / f"{model_name}_dic_and_emb.pkl"

        with open(input_path, "rb") as f:
            word2id, id2word, _, pca_embed, ica_embed = pickle.load(f)

        # ica
        ica_embed = pos_direct(ica_embed)
        skewness = np.mean(ica_embed**3, axis=0)
        skew_sort_idx = np.argsort(-skewness)
        ica_embed = ica_embed[:, skew_sort_idx]
        normed_ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)

        # pca
        normed_pca_embed = pca_embed / np.linalg.norm(pca_embed, axis=1, keepdims=True)

        model2normed_embed[model_name] = (normed_pca_embed, normed_ica_embed)

    def sci_notation(tick_val, pos):
        if tick_val != 0:
            exponent = int(np.log10(tick_val))
            base = tick_val / 10**exponent
            return f"{base:.0f}e{exponent}"
        else:
            return "0"

    ls = 30
    lsy = 28
    ts = 25

    fig, axes = plt.subplots(1, 4, figsize=(10 * len(dynamic_models), 8))
    # adjust ax
    plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.18, wspace=0.3)

    for i, model_name in enumerate(dynamic_models):
        ax = axes[i]
        pca_embed, ica_embed = model2normed_embed[model_name]

        n, dim = ica_embed.shape
        axis0_sorted_ica_idxs = np.argsort(-ica_embed, axis=0)
        axis0_sorted_pca_idxs = np.argsort(-pca_embed, axis=0)

        axis0_ica_ys = []
        axis0_pca_ys = []
        for axis_idx in tqdm(range(dim)):
            ica_idxs = axis0_sorted_ica_idxs[:, axis_idx]
            pca_idxs = axis0_sorted_pca_idxs[:, axis_idx]

            tmp_ica_ys = ica_embed[ica_idxs, axis_idx]
            tmp_pca_ys = pca_embed[pca_idxs, axis_idx]

            axis0_ica_ys.append(tmp_ica_ys)
            axis0_pca_ys.append(tmp_pca_ys)

        axis0_ica_ys = np.array(axis0_ica_ys)
        axis0_pca_ys = np.array(axis0_pca_ys)

        axis0_ica_ys_std = np.std(axis0_ica_ys, axis=0)
        axis0_pca_ys_std = np.std(axis0_pca_ys, axis=0)

        axis0_ica_ys = np.mean(axis0_ica_ys, axis=0)
        axis0_pca_ys = np.mean(axis0_pca_ys, axis=0)

        # color by nor
        ax.set_xlabel("Rank along Embeddings", fontsize=ls, labelpad=10)
        ax.set_ylabel("Average Component Value", fontsize=lsy)

        # ticks
        ax.set_xticks(range(0, n + 1, 10000))
        ax.set_yticks(np.arange(-0.2, 0.7, 0.2))

        # tick size
        ax.tick_params(axis="x", which="major", labelsize=ts)
        ax.tick_params(axis="y", which="major", labelsize=ts)

        # ylim
        ax.set_ylim(-0.3, 0.8)

        ax.plot(
            range(1, n + 1),
            axis0_ica_ys,
            label="ICA",
            linewidth=3,
            zorder=10,
            color="orange",
        )
        ax.plot(
            range(1, n + 1),
            axis0_pca_ys,
            label="PCA",
            linewidth=3,
            linestyle="--",
            color="blue",
        )

        # std
        ax.fill_between(
            range(1, n + 1),
            axis0_ica_ys - axis0_ica_ys_std,
            axis0_ica_ys + axis0_ica_ys_std,
            alpha=0.2,
            color="orange",
        )
        ax.fill_between(
            range(1, n + 1),
            axis0_pca_ys - axis0_pca_ys_std,
            axis0_pca_ys + axis0_pca_ys_std,
            alpha=0.2,
            color="blue",
        )

        ax.legend(fontsize=ls)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(sci_notation))

        title = model_name2title(model_name)
        ax.set_title(title, fontsize=ls)

    output_dir = Path("output/camera_ready_images")
    output_dir.mkdir(exist_ok=True, parents=True)
    img_path = output_dir / "sorted_along_embeddings_4models.png"
    logger.info(f"Saving to {img_path}")
    plt.savefig(img_path, dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(10 * len(dynamic_models), 8))
    # adjust ax
    plt.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.18, wspace=0.3)

    for i, model_name in enumerate(dynamic_models):
        ax = axes[i]
        pca_embed, ica_embed = model2normed_embed[model_name]

        n, dim = ica_embed.shape

        axis1_sorted_ica_idxs = np.argsort(-ica_embed, axis=1)
        axis1_sorted_pca_idxs = np.argsort(-pca_embed, axis=1)

        axis1_ica_ys = []
        axis1_pca_ys = []
        for i in tqdm(range(n)):
            ica_idxs = axis1_sorted_ica_idxs[i]
            pca_idxs = axis1_sorted_pca_idxs[i]

            tmp_ica_ys = ica_embed[i, ica_idxs]
            tmp_pca_ys = pca_embed[i, pca_idxs]

            axis1_ica_ys.append(tmp_ica_ys)
            axis1_pca_ys.append(tmp_pca_ys)

        axis1_ica_ys = np.array(axis1_ica_ys)
        axis1_pca_ys = np.array(axis1_pca_ys)

        axis1_ica_ys_std = np.std(axis1_ica_ys, axis=0)
        axis1_pca_ys_std = np.std(axis1_pca_ys, axis=0)

        axis1_ica_ys = np.mean(axis1_ica_ys, axis=0)
        axis1_pca_ys = np.mean(axis1_pca_ys, axis=0)

        # color by nor
        ax.set_xlabel("Rank along axes", fontsize=ls, labelpad=10)
        ax.set_ylabel("Average Component Value", fontsize=lsy)

        # ticks
        ax.set_yticks(np.arange(-0.2, 0.7, 0.2))

        # tick size
        ax.tick_params(axis="x", which="major", labelsize=ts)
        ax.tick_params(axis="y", which="major", labelsize=ts)

        # ylim
        ax.set_ylim(-0.3, 0.8)

        ax.plot(
            range(1, dim + 1),
            axis1_ica_ys,
            label="ICA",
            linewidth=3,
            zorder=10,
            color="orange",
        )
        ax.plot(
            range(1, dim + 1),
            axis1_pca_ys,
            label="PCA",
            linewidth=3,
            linestyle="--",
            color="blue",
        )

        # std
        ax.fill_between(
            range(1, dim + 1),
            axis1_ica_ys - axis1_ica_ys_std,
            axis1_ica_ys + axis1_ica_ys_std,
            alpha=0.2,
            color="orange",
        )
        ax.fill_between(
            range(1, dim + 1),
            axis1_pca_ys - axis1_pca_ys_std,
            axis1_pca_ys + axis1_pca_ys_std,
            alpha=0.2,
            color="blue",
        )

        ax.legend(fontsize=ls)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(sci_notation))

        title = model_name2title(model_name)
        ax.set_title(title, fontsize=ls)

    img_path = output_dir / "sorted_along_axes_4models.png"
    logger.info(f"Saving to {img_path}")
    plt.savefig(img_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
