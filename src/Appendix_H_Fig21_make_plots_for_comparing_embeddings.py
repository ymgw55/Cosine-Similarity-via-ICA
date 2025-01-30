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

    emb_path = 'output/embeddings/glove_dic_and_emb.pkl'
    with open(emb_path, 'rb') as f:
        word2id, id2word, embed, pca_embed, ica_embed = pkl.load(f)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]
    ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)
    n, _ = ica_embed.shape

    src_word = 'woman'
    for tgt_word in ['man', 'girl']:
        woman_i = word2id[src_word]
        man_i = word2id[tgt_word]

        woman_emb = ica_embed[woman_i]
        man_emb = ica_embed[man_i]

        female_axis = 77
        fs = 45
        ts = 34

        # scatter plot
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.scatter(woman_emb, man_emb, color='blue', s=50)

        woman_f = woman_emb[female_axis]
        man_f  = man_emb[female_axis]
        ax.scatter(woman_f, man_f, color='red', s=200, marker='s')
        # text "female"
        ax.text(woman_f, man_f+0.025, '[female]', fontsize=fs, color='red', ha='center')

        # y = x
        ax.plot([-1, 1], [-1, 1], color='black', linestyle='--')
        # x = 0 and y = 0
        ax.axhline(0, color='black')
        ax.axvline(0, color='black')
        # x label
        ax.set_xlabel(src_word, fontsize=fs)

        # y label
        ax.set_ylabel(tgt_word, fontsize=fs)

        # ticks
        ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])
        ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4])

        # ticks params
        ax.tick_params(labelsize=ts)

        # x and y lim
        ax.set_xlim(-0.2, 0.5)
        ax.set_ylim(-0.2, 0.5)

        # equal aspect ratio
        ax.set_aspect('equal')

        # adjust
        plt.subplots_adjust(left=0.18, right=0.999, top=0.99, bottom=0.14)
        output_dir = Path("output/camera_ready_images")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"plot_for_comparing_{src_word}_{tgt_word}.pdf"
        logger.info(f"Saving to {output_path}")
        plt.savefig(output_path)

if __name__ == '__main__':
    main()