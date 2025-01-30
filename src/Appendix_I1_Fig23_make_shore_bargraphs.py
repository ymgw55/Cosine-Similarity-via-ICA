import logging
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import pos_direct

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    bert_dump_path = \
        'data/embeddings/Universal-Geometry-with-ICA/bert-pca-ica-100000.pkl'
    with open(bert_dump_path, 'rb') as f:
        tokens_sents_embeds = pkl.load(f)
    id2token, bert_sents, _, pca_embed, ica_embed = tokens_sents_embeds

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]
    ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)
    n, dim = ica_embed.shape

    token2id = dict()
    for idx in range(n):
        token = id2token[idx]
        if token.startswith('shore_'):
            token2id[token] = idx

    axis_idxs = []
    for token in ['shore_0', 'shore_1', 'shore_2']:

        idx = token2id[token]
        emb_i = ica_embed[idx]
        sent = bert_sents[idx]
        logger.info(f'{token}: {sent}')
        top5_idx = np.argsort(-emb_i)[:4]

        for axis_idx in top5_idx:
            axis_idxs.append(axis_idx)
            top_ids = np.argsort(-ica_embed[:, axis_idx])[:10]
            top_tokens = [id2token[top_id] for top_id in top_ids]

    cos = np.dot(ica_embed[token2id['shore_0']], ica_embed[token2id['shore_1']])
    logger.info(f"cos(shore_0, shore_1): {cos:.3f}")
    cos = np.dot(ica_embed[token2id['shore_0']], ica_embed[token2id['shore_2']])
    logger.info(f"cos(shore_0, shore_2): {cos:.3f}")
    cos = np.dot(ica_embed[token2id['shore_1']], ica_embed[token2id['shore_2']])
    logger.info(f"cos(shore_1, shore_2): {cos:.3f}")

    axis_idxs = sorted(list(set(axis_idxs)))

    # we checked the top 10 words for each axis beforehand, and assigned labels
    labels = {
        60: 'sea',
        206: 'australia',
        249: 'district',
        302: 'control',
        337: 'causative verbs',
        343: 'numbers',
        353: 'orgnization',
        520: 'action',
        557: 'location',
        571: 'linking words',
    }

    rows = []
    for axis_idx in axis_idxs:
        top_ids = np.argsort(-ica_embed[:, axis_idx])[:10]
        top_tokens = [id2token[top_id] for top_id in top_ids]
        logger.info(f'{axis_idx+1}: {top_tokens}')
        
        row = dict()
        row['axis'] = axis_idx + 1
        for i, token in enumerate(top_tokens):
            row[f'top{i+1}'] = token
        row['meaning'] = labels[axis_idx]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv('output/Appendix_I1_Table14.csv', index=False)

    fig, axes = plt.subplots(1, 3, figsize=(30, 5))
    cmap = plt.get_cmap('rainbow', 10)
    colors = cmap(np.linspace(0, 1, 10))
    colors = colors[::-1]
    ax0 = axes[0]
    ax1 = axes[1]
    ax2 = axes[2]
    fs = 32
    ts = 25
    ls = 16

    shore0_emb = ica_embed[token2id['shore_0']]
    shore1_emb = ica_embed[token2id['shore_1']]
    shore2_emb = ica_embed[token2id['shore_2']]

    for idx, (ax, emb, word) in enumerate((
        [ax0, shore0_emb, 'shore_0'],
        [ax1, shore1_emb, 'shore_1'],
        [ax2, shore2_emb, 'shore_2'],
    )):
        ax.bar(np.arange(dim), emb, color='black', alpha=0.25)

        for cx, axis_idx in enumerate(axis_idxs):
            x = emb[axis_idx]
            dummy2 = np.zeros_like(emb)
            dummy2[axis_idx] = x
            ax.bar(np.arange(len(dummy2)), dummy2,
                   color=colors[cx], label=f'[{labels[axis_idx]}]: {x:.3f}', width=3)

        ax.set_title(word, fontsize=fs, pad=15)

        # y range
        ax.set_ylim(-0.15, 0.78)

        # y ticks
        ax.set_yticks(np.arange(-0., 0.5, 0.2))

        # x label
        ax.set_xlabel('Axis', fontsize=fs)
        # x ticks
        # ax.set_xticks(np.arange(0, dim+1, 200))
        ax.set_xticks(np.arange(100, dim+1, 100))

        if idx == 0:
            ax.set_ylabel('Normalized IC Value', fontsize=fs)

        ax.legend(fontsize=ls, loc='upper right', ncol=2)

        # ticks params
        ax.tick_params(labelsize=ts)

    # adjust
    plt.subplots_adjust(left=0.04, right=0.98, top=0.88,
                        bottom=0.18, wspace=0.1)

    output_dir = Path("output/camera_ready_images")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / "shore_bargraphs.pdf"
    logger.info(f"Saving to {save_path}")
    plt.savefig(save_path)


if __name__ == '__main__':
    main()