import logging
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from utils import pos_direct

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():

    emb_path = Path("output/embeddings/glove_dic_and_emb.pkl")
    logger.info(f"load GloVe embeddings from {emb_path}")
    with open(emb_path, "rb") as f:
        _, id2word, _, pca_embed, ica_embed = pkl.load(f)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]
    n, dim = ica_embed.shape
    logger.info(f"ica_embed.shape: {ica_embed.shape}")

    def heatmap(axis_idxs):
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(
            1, 2, figure=fig, width_ratios=[1] * 2, wspace=0.7, bottom=0.05, top=0.87
        )
        fig.subplots_adjust(left=0.15, right=0.87)
        cb_ax = fig.add_axes([0.9, 0.05, 0.03, 0.82])
        cb_ax.tick_params(labelsize=30)

        for idx, is_ica in enumerate([True, False]):
            ax = fig.add_subplot(gs[0, idx])

            if is_ica:
                embed = ica_embed
                cbar_ax = None
            else:
                embed = pca_embed
                cbar_ax = cb_ax

            normed_embed = embed / np.linalg.norm(embed, axis=1, keepdims=True)

            ls = 25
            ts = 22
            fs = 22

            wids = []
            for axis_idx in axis_idxs:
                axis_wids = np.argsort(-normed_embed[:, axis_idx])[:5]
                logger.info([id2word[wid] for wid in axis_wids])
                for wid in axis_wids:
                    wids.append(wid)

            show_emb = normed_embed[:, axis_idxs]

            g = sns.heatmap(
                show_emb[wids],
                yticklabels=[id2word[wid] for wid in wids],
                cmap="magma_r",
                ax=ax,
                vmin=-0.04,
                vmax=1.0,
                cbar_ax=cbar_ax,
                cbar=not is_ica,
            )
            g.tick_params(left=False, bottom=True, labelsize=ts)
            cbar = g.collections[0].colorbar
            if cbar:
                cbar.ax.tick_params(labelsize=fs)
            yticklabels = g.get_yticklabels()
            ax.set_yticklabels(
                yticklabels,
                rotation=0,
            )

            # xticks is labels
            ax.set_xticks(np.arange(len(axis_idxs)) + 0.5, minor=True)
            ax.set_xticklabels(
                [axis_idx + 1 for axis_idx in axis_idxs], rotation=0, fontsize=fs
            )
            ax.tick_params(axis="x", which="major", length=0)

            # title
            ax.set_title(
                f'Normalized {"ICA" if is_ica else "PCA"}-transformed\nEmbeddings',
                fontsize=ls,
                pad=20,
            )

        output_dir = Path("output/camera_ready_images")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / (
            "heatmap_comparing_ica_with_pca_" + "-".join(map(str, axis_idxs)) + ".pdf"
        )
        logger.info(f"save heatmap to {save_path}")
        plt.savefig(save_path)

    axis_idxs = [50 * i - 1 for i in range(1, 6)]

    heatmap(axis_idxs)


if __name__ == "__main__":
    main()
