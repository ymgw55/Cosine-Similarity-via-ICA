import logging
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import pos_direct

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():

    emb_path = "output/embeddings/glove_dic_and_emb.pkl"
    logger.info(f"load GloVe embeddings from {emb_path}")
    with open(emb_path, "rb") as f:
        word2id, id2word, embed, pca_embed, ica_embed = pkl.load(f)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]
    ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)
    n, dim = ica_embed.shape

    pca_embed = pca_embed / np.linalg.norm(pca_embed, axis=1, keepdims=True)

    def bargraphs(embed, src_word, tgt_word, labels, ylabel, path):
        src_i = word2id[src_word]
        src_emb_i = embed[src_i]
        # top5 components
        top5_idx = np.argsort(-src_emb_i)[:5]
        top5_idx = sorted(top5_idx)
        logger.info(
            f"{src_word}'s top 5 axis indices (1-index): {[idx+1 for idx in top5_idx]}"
        )
        for idx in top5_idx:
            top10_word_idx = np.argsort(-embed[:, idx])[:10]
            words = [id2word[i] for i in top10_word_idx]
            logger.info(f"top 10 words in axis {idx+1}: {words}")

        tgt_i = word2id[tgt_word]
        tgt_emb_i = embed[tgt_i]

        prod_i = src_emb_i * tgt_emb_i
        prod_word = f"{src_word} $\odot$ {tgt_word}"

        colors = ["red", "orange", "green", "deepskyblue", "blue"]
        fs = 32
        ts = 25
        ls = 14.5

        fig, axes = plt.subplots(1, 3, figsize=(30, 5))
        ax0 = axes[0]
        ax1 = axes[1]
        ax2 = axes[2]

        for idx, (ax, emb_i, word) in enumerate(
            (
                [ax0, src_emb_i, src_word],
                [ax1, tgt_emb_i, tgt_word],
                [ax2, prod_i, prod_word],
            )
        ):

            # print top 5 components values and their indices
            top5_idx = np.argsort(-emb_i)[:5]
            top5_idx = sorted(top5_idx, key=lambda x: -emb_i[x])
            for i, idx in enumerate(top5_idx):
                logger.info(
                    f"{word}'s top {i+1} axis index: {idx+1}, value: {emb_i[idx]:.3f}")
    
        
            ax.bar(np.arange(dim), emb_i, color="black", alpha=0.25)

            for cdx, axis_idx in enumerate(top5_idx):
                dummy2 = np.zeros_like(emb_i)
                x = emb_i[axis_idx]
                dummy2[axis_idx] = x
                ax.bar(
                    np.arange(len(dummy2)),
                    dummy2,
                    color=colors[cdx],
                    label=f"{labels[cdx]}: {x:.3f}",
                    width=1.5,
                )
            ax.set_title(f"{word}", fontsize=fs, pad=15)

            # y range
            ax.set_ylim(-0.25, 0.6)

            # y ticks
            ax.set_yticks(np.arange(-0.0, 0.5, 0.2))

            # x label
            ax.set_xlabel("Axis", fontsize=fs)
            # x ticks
            ax.set_xticks(np.arange(0, dim + 1, 100))

            if idx == 0:
                ax.set_ylabel(ylabel, fontsize=fs)
            ax.legend(fontsize=ls, loc="upper right")

            # ticks params
            ax.tick_params(labelsize=ts)

        # adjust
        plt.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.18, wspace=0.1)

        plt.savefig(path)

    output_dir = Path("output/camera_ready_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    top5_idx = np.argsort(ica_embed[word2id["ultraviolet"]])[::-1][:5]
    top5_idx = sorted(top5_idx)
    labels = ["chemistry", "biology", "space", "spectrum", "virology"]
    labels_with_axis_index = [f"{top5_idx[i]+1} [{labels[i]}]" for i in range(5)]
    bargraphs(
        ica_embed,
        "ultraviolet",
        "light",
        labels_with_axis_index,
        "Normalized IC Value",
        output_dir / "ultraviolet_and_light_bargraphs_ica.pdf",
    )
    top5_idx = np.argsort(pca_embed[word2id["ultraviolet"]])[::-1][:5]
    top5_idx = sorted(top5_idx)
    # labels = ["[PC80]", "[PC92]", "[PC152]", "[PC153]", "[PC222]"]
    labels = [f"[PC{top5_idx[i]+1}]" for i in range(5)]
    bargraphs(
        pca_embed,
        "ultraviolet",
        "light",
        labels,
        "Normalized PC Value",
        output_dir / "ultraviolet_and_light_bargraphs_pca.pdf",
    )


if __name__ == "__main__":
    main()
