import logging
import pickle as pkl
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


def main():

    emb_path = "output/embeddings/glove_dic_and_emb.pkl"
    with open(emb_path, "rb") as f:
        _, _, _, pca_embed, ica_embed = pkl.load(f)

    output_dir = Path("output/camera_ready_images")
    output_dir.mkdir(exist_ok=True, parents=True)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]
    ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)

    n, dim = ica_embed.shape
    pca_embed = pca_embed / np.linalg.norm(pca_embed, axis=1, keepdims=True)

    def sci_notation(tick_val, pos):
        if tick_val != 0:
            exponent = int(np.log10(tick_val))
            base = tick_val / 10**exponent
            return f"{base:.0f}e{exponent}"
        else:
            return "0"

    ls = 30
    lsy = 30
    ts = 25
    legend_s = 32

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

    fig, ax = plt.subplots(figsize=(8, 7))

    # adjust ax
    plt.subplots_adjust(left=0.18, right=0.99, top=0.97, bottom=0.13)

    # color by nor
    ax.set_xlabel("Rank along embeddings", fontsize=ls)
    ax.set_ylabel("Average Component Value", fontsize=lsy)

    # ticks
    ax.set_xticks(range(0, n + 1, 100000))
    ax.set_yticks(np.arange(-0.2, 0.7, 0.2))

    # tick size
    ax.tick_params(axis="x", which="major", labelsize=ts)
    ax.tick_params(axis="y", which="major", labelsize=ts)

    # ylim
    ax.set_ylim(-0.3, 0.7)

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

    ax.legend(fontsize=legend_s)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(sci_notation))
    img_path = output_dir / "sorted_along_embeddings.png"
    logger.info(f"Save to {img_path}")
    plt.savefig(img_path, dpi=150)
    plt.close()

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

    fig, ax = plt.subplots(figsize=(8, 7))

    # adjust ax
    plt.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.13)

    # color by nor
    ax.set_xlabel("Rank along axes", fontsize=ls)
    ax.set_ylabel("Average Component Value", fontsize=lsy)

    # ticks
    ax.set_yticks(np.arange(-0.2, 0.7, 0.2))

    # tick size
    ax.tick_params(axis="x", which="major", labelsize=ts)
    ax.tick_params(axis="y", which="major", labelsize=ts)

    # ylim
    ax.set_ylim(-0.3, 0.7)

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

    # ax.legend(fontsize=ls)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(sci_notation))

    img_path = output_dir / "sorted_along_axes.png"
    logger.info(f"Save to {img_path}")
    plt.savefig(img_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
