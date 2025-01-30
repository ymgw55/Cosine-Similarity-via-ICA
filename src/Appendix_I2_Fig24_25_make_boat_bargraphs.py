import logging
import pickle as pkl
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

output_dir = Path("output/camera_ready_images/boat_bargraphs")
output_dir.mkdir(exist_ok=True, parents=True)


def get_font_prop(lang):

    if lang == 'ja' or lang == 'zh':
        font_path = 'data/embeddings/Universal-Geometry-with-ICA/fonts/NotoSansCJKjp-Regular.otf'  # noqa
        font_prop = fm.FontProperties(fname=font_path, size=13)
    elif lang == 'hi':
        font_path = "data/embeddings/Universal-Geometry-with-ICA/fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf"  # noqa
        font_prop = fm.FontProperties(fname=font_path, size=13)
    else:
        font_prop = fm.FontProperties(size=13)
    return font_prop


def get_lang_name(lang):
    if lang == 'en':
        return 'English'
    elif lang == 'es':
        return 'Spanish'
    elif lang == 'fr':
        return 'French'
    elif lang == 'de':
        return 'German'
    elif lang == 'it':
        return 'Italian'
    elif lang == 'ru':
        return 'Russian'
    elif lang == 'ja':
        return 'Japanese'
    elif lang == 'zh':
        return 'Chinese'
    elif lang == 'ar':
        return 'Arabic'
    elif lang == 'ko':
        return 'Korean'
    elif lang == 'hi':
        return 'Hindi'
    else:
        raise ValueError(f'Unknown lang: {lang}')


def draw(src_word, lg, word, emb, top5_idx, labels, state):
    dim = len(emb)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(np.arange(dim), emb, color='black', alpha=0.25)

    colors = ['red', 'orange', 'green', 'deepskyblue', 'blue']
    fs = 30
    ts = 20
    ls = 15
    for cdx, axis_idx in enumerate(top5_idx):
        dummy2 = np.zeros_like(emb)
        x = emb[axis_idx]
        dummy2[axis_idx] = x
        ax.bar(np.arange(len(dummy2)), dummy2,
               color=colors[cdx], label=f'[{labels[cdx]}]: {x:.3f}', width=1.5)

    if lg == 'ar':
        title = word[::-1]
    else:
        title = word
    ax.set_title(title, fontsize=fs, fontproperties=get_font_prop(lg), pad=20)

    # y range
    ax.set_ylim(-0.25, 0.8)

    # y label
    if state == 'ica':
        ylabel = 'Normalized IC Value'
    else:
        ylabel = 'Normalized PC Value'
    ax.set_ylabel(ylabel, fontsize=fs)
    # y ticks
    ax.set_yticks(np.arange(-0.2, 0.61, 0.2))

    # x label
    ax.set_xlabel('Axis', fontsize=fs)
    # x ticks
    ax.set_xticks(np.arange(100, dim+1, 100))

    ax.legend(fontsize=ls, loc='upper right')

    # adjust
    plt.subplots_adjust(left=0.13, right=0.99, top=0.86, bottom=0.2)

    # ticks params
    ax.tick_params(labelsize=ts)
    path = output_dir / f'{state}_{lg}.pdf'
    logger.info(f"Saving to {path}")
    plt.savefig(path)


def main(state):
    dumped_path = Path(
        'data/embeddings/Universal-Geometry-with-ICA/en-es-ru-ar-hi-zh-ja/axis_matching_ica.pkl')  # noqa
    if dumped_path.exists():
        with open(dumped_path, 'rb') as f:
            _, _, normed_sorted_src_embed, normed_sorted_tgt_embeds, \
                src_id2word, tgt_id2words, \
                src_word2id, tgt_word2ids, sw2lang2tw = \
                pkl.load(f)

    src_word = 'boat'
    axis = np.argmax(normed_sorted_src_embed[src_word2id[src_word], :])
    tgt_list = ['es', 'ru', 'ar', 'hi', 'zh', 'ja']
    lg2w = dict()
    lg2w['en'] = src_word
    lg2emb = dict()
    lg2emb['en'] = normed_sorted_src_embed[src_word2id[src_word]]
    for tx, tgt_lang in enumerate(tgt_list):
        tws = sw2lang2tw[src_word][tgt_lang]
        tw_word2id = tgt_word2ids[tx]
        tmp_max = -10**10
        tmp_tw = None
        for tw_word in tws:
            tw_id = tw_word2id[tw_word]
            tgt_embed = normed_sorted_tgt_embeds[tx][tw_id]
            if tmp_max < tgt_embed[axis]:
                tmp_max = tgt_embed[axis]
                tmp_tw = tw_word

        lg2w[tgt_lang] = tmp_tw
        tw_id = tw_word2id[tmp_tw]
        lg2emb[tgt_lang] = normed_sorted_tgt_embeds[tx][tw_id]

    if state == 'pca':
        dumped_path = Path(
            '../embeddings/en-es-ru-ar-hi-zh-ja/axis_matching_pca.pkl')
        if dumped_path.exists():
            with open(dumped_path, 'rb') as f:
                _, _, normed_sorted_src_embed, normed_sorted_tgt_embeds, \
                    src_id2word, tgt_id2words, \
                    src_word2id, tgt_word2ids, sw2lang2tw = \
                    pkl.load(f)
        lg2emb = dict()
        lg2emb['en'] = normed_sorted_src_embed[src_word2id[src_word]]

        # lg2w is already set
        for tx, tgt_lang in enumerate(tgt_list):
            tw_word2id = tgt_word2ids[tx]
            tw_word = lg2w[tgt_lang]
            tw_id = tw_word2id[tw_word]
            lg2emb[tgt_lang] = normed_sorted_tgt_embeds[tx][tw_id]

    top5_idx = np.argsort(-lg2emb['en'])[:5]
    top5_idx = sorted(top5_idx)

    if state == 'ica':
        labels = ['[ship-and-sea]', '[cars]',
                  '[water]', '[multiple people]', '[races]']
    else:
        labels = [f'Aligned Axis{i+1}' for i in top5_idx]

    rows = []
    for cx, axis_idx in enumerate(top5_idx):

        top10_idxs = np.argsort(-normed_sorted_src_embed[:, axis_idx])[:10]

        assert len(top10_idxs) == 10
        top10_words = [src_id2word[i] for i in top10_idxs]

        row = dict()
        row['axis'] = axis_idx + 1
        for i, word in enumerate(top10_words):
            row[f'top{i+1}'] = word
        row['label'] = labels[cx]
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = f"output/Appendix_I2_Table16_{state}.csv"
    logger.info(f"Saving to {output_path}")
    df.to_csv(output_path, index=False)

    for lg in ['en'] + tgt_list:
        draw(src_word, lg, lg2w[lg], lg2emb[lg], top5_idx, labels, state)


if __name__ == '__main__':
    main('ica')
    main('pca')