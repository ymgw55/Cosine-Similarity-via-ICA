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
    with open(emb_path, "rb") as f:
        word2id, id2word, embed, pca_embed, ica_embed = pkl.load(f)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]
    ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)
    n, _ = ica_embed.shape

    def nearest_words(pred, topk=10, centering=False):
        # size: vocab_num
        cos = np.dot(ica_embed, pred)
        # top 10 index
        index = np.argpartition(-cos, topk)[:topk]
        top = index[np.argsort(-cos[index])]
        word_list = []
        sim_list = []
        for word_id in top:
            word = id2word[word_id]
            sim = cos[word_id]
            word_list.append(word)
            sim_list.append(sim)
        return word_list, sim_list

    axis_idxs = [12, 85, 134]
    axis_labels = {12: "food", 85: "animals", 134: "plants"}

    fig, ax = plt.subplots(1, 1, figsize=(44, 20))

    # horizontal line
    for i in range(1, 2 ** len(axis_idxs)):
        ax.axhline(i * 1.5, color="k", lw=0.5, alpha=0.5)

    fs = 32
    ms = 350
    ts = 40
    yts = ts
    ls = 50

    ticklabels = []
    max_sim = 0

    idx2color = {
        0: "red",
        1: "green",
        2: "blue",
        3: "orange",
        4: "magenta",
        5: "deepskyblue",
        6: "black",
    }

    word2color = {}
    color2marker = {
        "red": "o",
        "green": "*",
        "blue": "^",
        "orange": "D",
        "magenta": "v",
        "deepskyblue": "s",
        "black": "p",
    }

    for idx, axis_jdxs in enumerate(
        [
            [12],
            [85],
            [134],
            [12, 85],
            [12, 134],
            [85, 134],
            [12, 85, 134],
        ]
    ):
        query_embed = np.zeros(ica_embed.shape[1])
        for axis_jdx in axis_jdxs:
            query_embed[axis_jdx] = 1
        query_embed = query_embed / np.linalg.norm(query_embed)

        word_list, sim_list = nearest_words(query_embed, topk=10)
        logger.info(f"word_list: {word_list}")

        y = 1.5 * (2 ** len(axis_idxs) - 1 - idx)
        for rank, (word, x) in enumerate(zip(word_list, sim_list)):
            max_sim = max(max_sim, x)

            # adjust some positions manually
            sign = 1
            if idx == 0 and word in ("baked", "butter"):
                sign = -1
            if idx == 1 and word in ("squirrels"):
                sign = -1
            if idx == 2 and word in ("vines", "saplings", "tree"):
                sign = -1
            if idx == 3 and word in ("fish", "grilled"):
                sign = -1
            if idx == 4 and word in ("mango", "cabbage"):
                sign = -1
            if idx == 5 and word in ("trees", "squirrels"):
                sign = -1
            if idx == 6 and word in ("fruits", "carrots"):
                sign = -1

            dx = 0

            # adjust some positions manually
            if idx == 1 and word in ("deer"):
                dx = 0.002

            if idx == 4 and word in ("cabbage"):
                dx = -0.002

            if idx == 5 and word in ("species"):
                dx = 0.002

            if word not in word2color:
                word2color[word] = idx2color[idx]

            color = word2color[word]

            ax.scatter(
                x,
                y,
                s=ms,
                c=color,
                edgecolors="k",
                linewidths=0.5,
                zorder=10,
                marker=color2marker[color],
            )
            ax.text(
                x + (0.008) * sign + dx,
                y + 0.8 * sign,
                f"{word} ({rank + 1})",
                fontsize=fs,
                ha="center",
                va="center",
                rotation=45,
                color=color,
            )

        ticklabel = (
            " \n& ".join([f"[{axis_labels[axis_jdx]}]" for axis_jdx in axis_jdxs]) + " "
        )
        ticklabels.append(ticklabel)

    ax.set_yticks(1.5 * np.arange(1, len(ticklabels) + 1))
    ax.set_yticklabels(ticklabels[::-1], fontsize=yts)

    for i, ticklabel in enumerate(list(ax.get_yticklabels())[::-1]):
        ticklabel.set_color(idx2color.get(i, "black"))

    ax.set_ylim(0, 1.5 * (len(ticklabels) + 1))
    ax.tick_params(labelsize=ts)
    ax.tick_params(axis="y", which="major", length=0)
    ax.set_xlabel("Cosine Similarity", fontsize=ls)

    # xlim
    logger.info(f"max_sim: {max_sim}")
    ax.set_xlim(0.485, 0.74)

    # adjust
    plt.subplots_adjust(left=0.08, right=0.99, top=0.99, bottom=0.08)

    output_dir = Path("output/camera_ready_images")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / "plot_for_food_animals_plants.pdf"
    logger.info(f"Saving to {save_path}")
    plt.savefig(save_path)


if __name__ == "__main__":
    main()
