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
    with open(emb_path, 'rb') as f:
        word2id, id2word, embed, pca_embed, ica_embed = pkl.load(f)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]
    ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)
    n, dim = ica_embed.shape

    pca_embed = pca_embed / np.linalg.norm(pca_embed, axis=1, keepdims=True)

    src_word = 'ultraviolet'

    # freeze seed
    np.random.seed(0)

    output_dir = Path("output/camera_ready_images")
    output_dir.mkdir(exist_ok=True, parents=True)

    def show_histogram(embed, method):
    
        ls = 22
        ts = 15
        ls2 = 8
        colors = ['red', 'orange', 'green', 'deepskyblue', 'blue']

        src_i = word2id[src_word]
        src_emb = embed[src_i]
        # top5 components
        top5_idx = np.argsort(-src_emb)[:5]
        top5_idx = sorted(top5_idx)
        logger.info(f'top5_idx: {[idx + 1 for idx in top5_idx]}')
        for idx in top5_idx:
            top10_word_idx = np.argsort(-embed[:, idx])[:10]
            words = [id2word[i] for i in top10_word_idx]
            logger.info(f'top10 words for component {idx+1}: {words}')

        tgt_words = ['salts', 'proteins', 'spacecraft', 'light', 'virus']
        tgt_embs = [embed[word2id[tgt_word]] for tgt_word in tgt_words]

        # random 10,000 samples
        samples = np.random.choice(n, 10000)
        src_emb_samples = embed[samples, :]

        # random 10,000 samples
        samples = np.random.choice(n, 10000)
        tgt_emb_samples = embed[samples, :]

        prods = src_emb_samples * tgt_emb_samples
        cossims = np.sum(prods, axis=1)

        logger.info(f'inverse of variance: {1/np.var(cossims)}')

        mu = 0
        xpad = 10

        sigma2 = 1 / dim
        siginv = dim
        color = 'gray'
        bins = np.linspace(-0.6, 0.6, 100)
        x = np.linspace(-0.6, 0.6, 100)
        y = np.exp(-(x-mu)**2 / (2*sigma2)) / np.sqrt(2*np.pi*sigma2)
        pad = 25
        label = r'$\mathcal{N}(0, 1/' + str(siginv) + r')$'

        fig, ax = plt.subplots()
        ds, _, _ = ax.hist(cossims, bins=bins,
                alpha=0.25, color='black', density=True, label='Random Samples')
        max_y = np.max(ds)

        ax.plot(x, y, label=label, color='black', linewidth=1)

        ax.set_xlabel('Cosine Similarity', fontsize=ls,
                labelpad=xpad)
        ax.set_ylabel('Density', fontsize=ls, labelpad=pad)
        ax.tick_params(labelsize=ts)

        for idx in range(5):
            tgt_word = tgt_words[idx]
            tgt_emb = tgt_embs[idx]
            cos = np.dot(src_emb, tgt_emb)
            color = colors[idx]
            ax.vlines(cos, 0, max_y, color=color, linestyles='-', linewidth=2,
                      label=f'cos({src_word}, {tgt_word}): ${cos:.3f}$')

        ax.legend(fontsize=ls2, loc='upper left')

        ax.set_title('ICA and PCA', fontsize=ls)
        save_path = f'cossim_histogram_{method}.png'
        logger.info(f'Saving to {save_path}')

        plt.subplots_adjust(left=0.15, right=0.99, top=0.92, bottom=0.16, wspace=0.1)
        plt.savefig(f'output/camera_ready_images/histograms/{save_path}', dpi=150)
        plt.close()

    show_histogram(ica_embed, 'ica')



if __name__ == '__main__':
    main()