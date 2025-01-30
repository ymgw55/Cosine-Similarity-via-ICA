import logging
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import k0

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
    logger.info(f"dimesion of embeddings: {dim}")

    pca_embed = pca_embed / np.linalg.norm(pca_embed, axis=1, keepdims=True)

    src_word = "ultraviolet"
    tgt_word = "light"

    # freeze seed
    np.random.seed(0)

    output_dir = Path("output/camera_ready_images/histograms")
    output_dir.mkdir(exist_ok=True, parents=True)

    def show_histogram(embed, method, labels):
        ls = 22
        ts = 15
        ls2 = 8
        colors = ["red", "orange", "green", "deepskyblue", "blue"]

        src_i = word2id[src_word]
        src_emb = embed[src_i]
        # top5 components
        top5_idx = np.argsort(-src_emb)[:5]
        top5_idx = sorted(top5_idx)
        logger.info(
            f"{src_word}'s top 5 axis indices (1-index): {[idx+1 for idx in top5_idx]}"
        )
        for idx in top5_idx:
            top10_word_idx = np.argsort(-embed[:, idx])[:10]
            words = [id2word[i] for i in top10_word_idx]
            logger.info(f"top 10 words in axis {idx+1}: {words}")

        tgt_i = word2id[tgt_word]
        tgt_emb = embed[tgt_i]

        prod_emb = src_emb * tgt_emb

        # random 10,000 samples
        samples = np.random.choice(n, 10000)
        src_emb_samples = embed[samples, :]
        component_samples = src_emb_samples.flatten()
        logger.info(f"number of samples: {len(samples)}")
        logger.info(f"number of components (flattened): {len(component_samples)}")
        logger.info(f"variance of component values: {np.var(component_samples):.6f}")
        logger.info(
            f"reciprocal of variance of component values: {1/np.var(component_samples):.3f}"
        )

        # random 10,000 samples
        samples = np.random.choice(n, 10000)
        tgt_emb_samples = embed[samples, :]

        prod_emb_samples = src_emb_samples * tgt_emb_samples
        prods = prod_emb_samples.flatten()
        logger.info(f"number of samples: {len(samples)}")
        logger.info(f"number of products (flattened): {len(prods)}")
        logger.info(f"variance of product values: {np.var(prods):.6f}")
        logger.info(f"reciprocal of variance of product values: {1/np.var(prods):.3f}")

        mu = 0

        for data_type, data, magnified in [
            ("Component", component_samples, False),
            ("Component-wise Product", prods, False),
            ("Component-wise Product", prods, True),
        ]:
            xpad = 10

            # plot normal distribution
            if data_type == "Component":
                sigma2 = 1 / dim
                siginv = dim
                color = "gray"
                bins = np.linspace(-0.6, 0.6, 100)
                x = np.linspace(-0.6, 0.6, 100)
                y = np.exp(-((x - mu) ** 2) / (2 * sigma2)) / np.sqrt(
                    2 * np.pi * sigma2
                )
                pad = 25
                label = r"$\mathcal{N}(0, 1/" + str(siginv) + r")$"
            else:
                if magnified:
                    bins = np.linspace(-0.01, 0.01, 500)
                    x = np.linspace(-0.01, 0.01, 1000)
                else:
                    bins = np.linspace(-0.3, 0.3, 100)
                    x = np.linspace(-0.3, 0.3, 100)

                color = "gray"
                y = k0(abs(x) * dim) * dim / np.pi
                pad = 10
                label = r"$300\mathcal{K}_0(300|x|)/\pi$"

            fig, ax = plt.subplots()
            ds, _, _ = ax.hist(
                data,
                bins=bins,
                alpha=0.25,
                color="black",
                density=True,
                label="Random Samples",
            )
            max_y = np.max(ds)

            ax.plot(x, y, label=label, color="black", linewidth=1)

            ax.set_xlabel(f"{data_type} Value", fontsize=ls, labelpad=xpad)
            ax.set_ylabel("Density", fontsize=ls, labelpad=pad)
            ax.tick_params(labelsize=ts)

            if data_type == "Component":
                for idx in range(5):
                    axis_idx = top5_idx[idx]
                    label = labels[idx]
                    value = src_emb[axis_idx]
                    color = colors[idx]
                    ax.vlines(
                        value,
                        0,
                        max_y,
                        color=color,
                        linestyles="-",
                        linewidth=2,
                        label=f"{src_word} [{label}]: ${value:.3f}$",
                    )

                for idx in range(5):
                    axis_idx = top5_idx[idx]
                    label = labels[idx]
                    value = tgt_emb[axis_idx]
                    color = colors[idx]
                    ax.vlines(
                        value,
                        0,
                        max_y,
                        color=color,
                        linestyles="--",
                        linewidth=2,
                        label=f"{tgt_word} [{label}]: ${value:.3f}$",
                    )
            else:
                if not magnified:
                    for idx in range(5):
                        axis_idx = top5_idx[idx]
                        label = labels[idx]
                        value = prod_emb[axis_idx]
                        color = colors[idx]
                        ax.vlines(
                            value,
                            0,
                            max_y,
                            color=color,
                            linestyles="-",
                            linewidth=1,
                            label=f"[{label}]: ${value:.3f}$",
                        )

            if data_type == "Component":
                ax.legend(fontsize=ls2, loc="upper left")
            else:
                ax.legend(fontsize=int(1.5 * ls2), loc="upper left")

            ax.set_title(method.upper(), fontsize=ls)
            magnified_str = "magnified" if magnified else "full"
            save_path = (
                output_dir
                / f'normalized_{"_".join(data_type.split())}_values_histogram_{method}_{magnified_str}.png'
            )
            logger.info(f"save histogram to {save_path}")

            plt.subplots_adjust(
                left=0.15, right=0.99, top=0.92, bottom=0.16, wspace=0.1
            )
            plt.savefig(save_path, dpi=150)
            plt.close()

    labels = ["chemistry", "biology", "space", "spectrum", "virology"]
    show_histogram(ica_embed, "ica", labels)

    labels = ["PC80", "PC92", "PC152", "PC153", "PC222"]
    show_histogram(pca_embed, "pca", labels)


if __name__ == "__main__":
    main()
