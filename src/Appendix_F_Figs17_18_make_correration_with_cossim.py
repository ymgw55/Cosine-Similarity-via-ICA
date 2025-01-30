import logging
import pickle as pkl
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from web.datasets.similarity import (fetch_MEN, fetch_MTurk, fetch_RG65,
                                     fetch_RW, fetch_SimLex999, fetch_WS353)
from web.embedding import Embedding
from web.evaluate import return_cossim_and_toppsum
from web.vocabulary import OrderedVocabulary

from utils import pos_direct

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MyEmbedding(Embedding):
    # override
    def __init__(self, vocab, vectors, p):
        super().__init__(vocab, vectors)
        self.p = p

    @staticmethod
    def from_words_and_vectors(words, vectors, p):
        vocab = OrderedVocabulary(words)
        return MyEmbedding(vocab, vectors, p)


def main():
    # seed
    np.random.seed(0)

    similarity_tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "WS353R": fetch_WS353(which="relatedness"),
        "WS353S": fetch_WS353(which="similarity"),
        "SimLex999": fetch_SimLex999(),
        "RW": fetch_RW(),
        "RG65": fetch_RG65(),
        "MTurk": fetch_MTurk(),
    }

    emb_path = "output/embeddings/glove_dic_and_emb.pkl"
    with open(emb_path, "rb") as f:
        word2id, id2word, embed, pca_embed, ica_embed = pkl.load(f)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]

    ps = [1, 2, 5, 10, 20, 50, 100, 200, 300]

    emb2cossim = {"pca": {}, "ica": {}}
    emb2toppsum = {"pca": {}, "ica": {}}

    # store cossim and toppsum
    for emb_type in ["pca", "ica"]:
        if emb_type == "pca":
            embed = pca_embed
        elif emb_type == "ica":
            embed = ica_embed

        for p in ps:
            w = MyEmbedding.from_words_and_vectors(id2word, embed, p)

            all_cossims = []
            all_toppsums = []

            # sim tasks
            for task_name, task in similarity_tasks.items():
                cossims, scores = return_cossim_and_toppsum(w, task.X, task.y)
                all_cossims += cossims
                all_toppsums += scores

            all_cossims = np.array(all_cossims)
            all_toppsums = np.array(all_toppsums)

            emb2cossim[emb_type][p] = all_cossims
            emb2toppsum[emb_type][p] = all_toppsums

    ls = 25
    ts = 20
    ls2 = 25

    output_dir = Path("output/camera_ready_images")
    output_dir.mkdir(exist_ok=True, parents=True)

    # plot scatter plots
    for p in [1, 10, 100]:
        fig, ax = plt.subplots(figsize=(8, 7))
        plt.subplots_adjust(left=0.15, right=0.93, top=0.99, bottom=0.14)

        xs_ica = emb2toppsum["ica"][p]
        ys_ica = emb2cossim["ica"][p]
        ax.scatter(
            xs_ica,
            ys_ica,
            c="orange",
            label="ICA",
            marker="o",
            s=10,
            alpha=0.5,
            zorder=10,
        )

        xs_pca = emb2toppsum["pca"][p]
        ys_pca = emb2cossim["pca"][p]
        ax.scatter(xs_pca, ys_pca, c="blue", label="PCA", marker="x", s=10, alpha=0.5)

        if p == 1:
            ax.set_xlabel(f"Top {p} Component-wise Product", fontsize=ls)
        else:
            ax.set_xlabel(
                f"Sum of Top {p} Component-wise Products", fontsize=ls, labelpad=10
            )
        ax.set_ylabel("Cosine Similarity", fontsize=ls)
        ax.legend(loc="lower right", fontsize=ls2)
        ax.tick_params(labelsize=ts)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.25, 1.05])
        save_path = output_dir / f"correration_with_cossim_p{p}.png"
        logger.info(f"Saving to {save_path}")
        plt.savefig(save_path, dpi=150)
        plt.close()

    # plot correlation with cosine similarity
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.subplots_adjust(left=0.13, right=0.99, top=0.92, bottom=0.13)
    rs_ica = [
        scipy.stats.spearmanr(emb2toppsum["ica"][p], emb2cossim["ica"][p])[0]
        for p in ps
    ]
    rs_pca = [
        scipy.stats.spearmanr(emb2toppsum["pca"][p], emb2cossim["pca"][p])[0]
        for p in ps
    ]

    for r_ica, r_pca, p in zip(rs_ica, rs_pca, ps):
        logger.info(f"p={p}, ICA: {r_ica:.3f}, PCA: {r_pca:.3f}")

    linestyle = "-"
    color = "orange"
    label = "ICA"
    ax.plot(
        ps,
        rs_ica,
        label=label,
        marker="o",
        linewidth=3,
        markersize=10,
        linestyle=linestyle,
        color=color,
        zorder=10,
    )

    linestyle = "--"
    color = "blue"
    label = "PCA"
    ax.plot(
        ps,
        rs_pca,
        label=label,
        marker="o",
        linewidth=3,
        markersize=10,
        linestyle=linestyle,
        color=color,
    )

    ax.set_xlabel("Number of Non-Zero Axes", fontsize=ls)
    ax.set_ylabel(r"Spearman's $\rho$", fontsize=ls)
    ax.set_xscale("log")
    ax.legend(loc="lower right", fontsize=ls2)
    ax.set_title("Sum of Top $p$ Products and Cosine Similarity", fontsize=ls, pad=10)
    ax.tick_params(labelsize=ts)

    save_path = output_dir / "correration_with_cossim_spearmanr.pdf"
    logger.info(f"Saving to {save_path}")
    plt.savefig(save_path)


if __name__ == "__main__":
    main()
