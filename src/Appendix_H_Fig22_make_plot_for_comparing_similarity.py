import logging
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text

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
    n, dim = ica_embed.shape

    src_word = 'woman'

    src_i = word2id[src_word]
    src_emb = ica_embed[src_i]

    # [female] axis
    axis_idx = 77
    female_embs = ica_embed[:, axis_idx]
    top5_idx = np.argsort(-female_embs)[:5]
    top5_idx = sorted(top5_idx)
    logger.info(
        f"aixs_idx: {axis_idx + 1}, top5 words: {[id2word[i] for i in top5_idx]}"
    )

    zero_embed = src_emb.copy()
    zero_embed[axis_idx] = 0
    xs = np.dot(ica_embed, zero_embed)
    ys = ica_embed[:, axis_idx]

    fig, ax = plt.subplots()

    ax.set_xlim(xs.min()*1.1, xs.max()*1.1)
    ax.set_ylim(min(0, ys.min())*1.1, ys.max()*1.1)
    xmin = xs.min()*1.1
    xmax = xs.max()*1.1
    ymin = min(0, ys.min())*1.1
    ymax = ys.max()*1.1

    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(x, y)
    src_y = ica_embed[src_i, axis_idx]

    Z = np.zeros((100, 100))

    for i in range(100):
        for j in range(100):
            Z[i, j] = min(1.0, X[i, j] + Y[i, j] * src_y)
            Z[i, j] = max(-1.0, Z[i, j])

    alpha = 0.6
    CS = ax.contour(X, Y, Z, levels=[0.0, 0.4, 0.8], alpha=alpha, colors='black')
    ax.clabel(CS, inline=True, fontsize=12, colors='black')

    fs = 12
    ts = 14
    ls = 20

    sc = 'lime'
    lc = 'gray'

    lw = 0.5
    ec = 'black'

    ax.scatter(xs, ys, c=sc, s=5)

    ax.set_xlabel(r'$\hat{\mathbf{s}}_{i_\mathrm{woman}\ominus\ell_{[\mathrm{female}]}}{}^\top \hat{\mathbf{s}}_i$',  # noqa
                  fontsize=ls)
    ax.set_ylabel(r'$\hat{s}^{(\ell_{[\mathrm{female}]})}_i$', fontsize=ls)
    texts = []

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

    org_cand_words, _ = nearest_words(src_emb, topk=100)

    ablating_emb = src_emb.copy()
    ablating_emb = ablating_emb / np.linalg.norm(ablating_emb)
    ablating_emb[axis_idx] = 0
    ablating_cand_words, _ = nearest_words(ablating_emb, topk=100)
    common_words = set(org_cand_words[1:11]) & set(ablating_cand_words[1:11])

    for topi, top_word in enumerate(org_cand_words[1:11]):
        if top_word not in common_words:

            marker = '^'
            tc = 'red'
        else:
            marker = 'D'
            tc = 'black'

        ax.scatter(xs[word2id[top_word]], ys[word2id[top_word]], c=tc, s=45, zorder=10,
                   marker=marker, edgecolors=ec, linewidths=lw)
        texts.append(ax.text(xs[word2id[top_word]], ys[word2id[top_word]],
                     fontsize=fs, s=top_word, c=tc, ha='center'))

    for topi, top_word in enumerate(ablating_cand_words[1:11]):
        if top_word not in common_words:

            marker = 'v'
            tc = 'blue'

            ax.scatter(xs[word2id[top_word]], ys[word2id[top_word]], c=tc, s=45,
            zorder=10, marker=marker, edgecolors=ec, linewidths=lw)
            texts.append(ax.text(xs[word2id[top_word]], ys[word2id[top_word]], 
            fontsize=fs, s=top_word, c='blue', ha='center'))

    # star shape
    ax.scatter(xs[src_i], ys[src_i], c='orange', s=45, zorder=10, marker='*')
               
    ax.text(xs[src_i]+0.04, ys[src_i]+0.03, 'woman', fontsize=int(fs*1.2), 
            c='white', ha='right', bbox=dict(facecolor='orange', alpha=0.9,
            boxstyle='round, pad=0.1', edgecolor='black', linewidth=lw))

    # adjust text
    adjust_text(texts,
                arrowprops=dict(arrowstyle="->", color='k', lw=0.5), 
                force_pull=(0.1, 0.1),
                force_text=(0.05, 0.05),
                force_explode=(0.005, 0.005),
                expand_axes=False,)

    # tick size
    ax.tick_params(labelsize=ts)

    # tick label
    ax.set_xticks(np.arange(-0.2, 0.9, 0.2))
    ax.set_yticks(np.arange(-0.2, 0.6, 0.2))

    # x-axis
    ax.axhline(0, color=lc, lw=1.5)
    # y-axis
    ax.axvline(0, color=lc, lw=1.5)

    # add vline
    word_i = word2id['women']
    word_emb = ica_embed[word_i]
    dot = np.dot(word_emb, zero_embed)
    ax.axvline(dot, color='deepskyblue', lw=2, linestyle='--', alpha=alpha)

    # draw k = x + y * src_y
    # the line pass daughter
    word_i = word2id['daughter']
    word_emb = ica_embed[word_i]
    dot = np.dot(word_emb, src_emb)
    a = - 1 / src_y
    b = dot / src_y
    ax.plot([xmin, xmax], [a*xmin+b, a*xmax+b], color='magenta',
            lw=2, linestyle='-.', alpha=alpha)
    plt.subplots_adjust(left=0.17, right=0.99, top=0.99, bottom=0.15)

    output_dir = Path("output/camera_ready_images")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / 'plot_for_comparing_similarity.png'
    logger.info(f"Saving to {output_path}")
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    main()