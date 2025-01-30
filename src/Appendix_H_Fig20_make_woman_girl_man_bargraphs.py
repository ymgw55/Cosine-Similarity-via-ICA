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


def main(src_word="woman", tgt1_word="girl", tgt2_word="man"):

    emb_path = "output/embeddings/glove_dic_and_emb.pkl"
    with open(emb_path, "rb") as f:
        word2id, id2word, embed, pca_embed, ica_embed = pkl.load(f)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]
    ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)
    n, dim = ica_embed.shape

    src_i = word2id[src_word]
    src_emb = ica_embed[src_i]

    female_axis = 77
    female_embs = ica_embed[:, female_axis]
    top5_idx = np.argsort(-female_embs)[:5]
    top5_idx = sorted(top5_idx)
    logger.info(f"Top 5 words of the [female] axis: {[id2word[i] for i in top5_idx]}")

    query_emb = src_emb.copy()
    query_emb[female_axis] = 0

    tgt1_i = word2id[tgt1_word]
    tgt1_emb = ica_embed[tgt1_i]

    tgt2_i = word2id[tgt2_word]
    tgt2_emb = ica_embed[tgt2_i]

    fs = 32
    ts = 25
    ls = 25

    fig, axes = plt.subplots(1, 3, figsize=(30, 5))
    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]

    for idx, (ax, emb, word) in enumerate(
        (
            [ax0, src_emb, src_word],
            [ax1, tgt1_emb, tgt1_word],
            [ax2, tgt2_emb, tgt2_word],
        )
    ):
        ax.bar(np.arange(dim), emb, color="blue", alpha=0.5)
        dummy2 = np.zeros_like(emb)
        x = emb[female_axis]
        dummy2[female_axis] = x
        ax.bar(
            np.arange(len(dummy2)),
            dummy2,
            color="red",
            label=f"[female]: {x:.3f}",
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
        ax.set_xticks(np.arange(100, dim + 1, 100))

        if idx == 0:
            ax.set_ylabel("Normalized IC Value", fontsize=fs)
        ax.legend(fontsize=ls, loc="upper right")

        # ticks params
        ax.tick_params(labelsize=ts)

    # adjust
    plt.subplots_adjust(left=0.04, right=0.99, top=0.88, bottom=0.18, wspace=0.1)
    output_dir = Path("output/camera_ready_images")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{src_word}_{tgt1_word}_{tgt2_word}_bargraphs.pdf"
    logger.info(f"Saving to {output_path}")
    plt.savefig(output_path)


if __name__ == "__main__":
    main()
