import logging
import pickle
import pickle as pkl
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import pos_direct

warnings.filterwarnings('ignore')

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(ica=False):

    dynamic_models = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "EleutherAI-pythia-160m",
    ]

    input_dir = Path("output/data_for_ultraviolet_and_light_bargraphs/")
    input_path = input_dir / "token_and_model2new_idx_and_sentence_light.pkl"
    with open(input_path, "rb") as f:
        light_dict = pkl.load(f)
    input_path = input_dir / "token_and_model2new_idx_and_sentence_ultraviolet.pkl"
    with open(input_path, "rb") as f:
        ultraviolet_dict = pkl.load(f)

    unew_idx2token_and_model = {}
    for (token, model_name), (new_idx, sentence) in ultraviolet_dict.items():
        unew_idx2token_and_model[new_idx] = (token, model_name)
    lnew_idx2token_and_model = {}
    for (token, model_name), (new_idx, sentence) in light_dict.items():
        lnew_idx2token_and_model[new_idx] = (token, model_name)

    # Load embeddings
    input_dir = Path("output/embeddings/")

    # dynamic
    new_idx2dict_list = defaultdict(list)
    model2normed_embed = {}
    modelid2word = {}
    for model_name in dynamic_models:
        logger.info(model_name)

        input_path = input_dir / f"{model_name}_dic_and_emb.pkl"

        with open(input_path, "rb") as f:
            word2id, id2word, _, pca_embed, ica_embed = pickle.load(f)
            modelid2word[model_name] = id2word

        if ica:
            embed = ica_embed
            embed = pos_direct(embed)
            skewness = np.mean(embed**3, axis=0)
            skew_sort_idx = np.argsort(-skewness)
            embed = embed[:, skew_sort_idx]
        else:
            embed = pca_embed

        normed_embed = embed / np.linalg.norm(embed, axis=1, keepdims=True)
        model2normed_embed[model_name] = normed_embed

        bert_prefix = "##"
        other_prefix = "Ġ"

        ultraviolet_list = []
        light_list = []
        if model_name == "bert-base-uncased":
            for word, id in word2id.items():
                if word.startswith("ultraviolet_") or word.startswith(
                    bert_prefix + "ultraviolet_"
                ):
                    if (word, model_name) not in ultraviolet_dict:
                        continue
                    new_idx, sentence = ultraviolet_dict[(word, model_name)]
                    ultraviolet_list.append((id, new_idx, sentence))
            for word, id in word2id.items():
                if word.startswith("light_") or word.startswith(bert_prefix + "light_"):
                    if (word, model_name) not in light_dict:
                        continue
                    new_idx, sentence = light_dict[(word, model_name)]
                    light_list.append((id, new_idx, sentence))
        else:
            for word, id in word2id.items():
                if word.startswith("ultraviolet_") or word.startswith(
                    other_prefix + "ultraviolet_"
                ):
                    if (word, model_name) not in ultraviolet_dict:
                        continue
                    new_idx, sentence = ultraviolet_dict[(word, model_name)]
                    ultraviolet_list.append((id, new_idx, sentence))
            for word, id in word2id.items():
                if word.startswith("light_") or word.startswith(
                    other_prefix + "light_"
                ):
                    if (word, model_name) not in light_dict:
                        continue
                    new_idx, sentence = light_dict[(word, model_name)]
                    light_list.append((id, new_idx, sentence))

        for uid, unew_idx, usent in ultraviolet_list:
            for lid, lnew_idx, lsent in light_list:
                uword = id2word[uid]
                lword = id2word[lid]
                norm_u = normed_embed[uid]
                norm_l = normed_embed[lid]
                cos = np.dot(norm_u, norm_l)

                new_idx2dict_list[(unew_idx, lnew_idx)].append(
                    {
                        "model_name": model_name,
                        "uid": uid,
                        "uword": uword,
                        "usent": usent,
                        "norm_u": norm_u,
                        "lid": lid,
                        "lword": lword,
                        "lsent": lsent,
                        "norm_l": norm_l,
                        "cos": cos,
                    }
                )

    for k, v in new_idx2dict_list.items():
        assert len(v) == len(dynamic_models)

    max_avg_key = None
    max_avg_cos = 0
    for k, v in new_idx2dict_list.items():
        avg_cos = sum([x["cos"] for x in v]) / len(v)
        if avg_cos > max_avg_cos:
            max_avg_cos = avg_cos
            max_avg_key = k

    max_avg_v = new_idx2dict_list[max_avg_key]
    logger.info(f"max_avg_key: {max_avg_key}, max_avg_cos: {max_avg_cos}")
    for dic in max_avg_v:
        for k, v in dic.items():
            if k in ["norm_u", "norm_l"]:
                continue
            logger.info(f"{k}: {v}")
        logger.info("=" * 50)

    num_rows = 4
    fig, axs = plt.subplots(num_rows, 3, figsize=(30, 4*num_rows))
    colors = ['red', 'orange', 'green', 'deepskyblue', 'blue']
    title_fs = 32
    label_fs = 32
    ylabel_fs = 24
    legend_fs = 15
    ts = 25

    legend_loc = 'upper right'

    if ica:
        model2labels = {
            'bert-base-uncased': ['biology', 'environment', 'topics', 'color', 'broadcasting'],
            'roberta-base': ['environment', 'biology', 'weekdays', 'networks', 'adjectives'],
            'gpt2': ['biology', 'elements', 'light', 'connectors', 'internet'],
            'EleutherAI-pythia-160m': ['light', 'elements', 'danger', 'fragments', 'exceptional'],
        }
    else:
        model2labels = {
            'bert-base-uncased': ['dummy', 'dummy', 'dummy', 'dummy', 'dummy'],
            'roberta-base': ['dummy', 'dummy', 'dummy', 'dummy', 'dummy'],
            'gpt2': ['dummy', 'dummy', 'dummy', 'dummy', 'dummy'],
            'EleutherAI-pythia-160m': ['dummy', 'dummy', 'dummy', 'dummy', 'dummy'],
        }


    model2ylabel = {
        'bert-base-uncased': 'BERT-base',
        'roberta-base': 'RoBERTa-base',
        'gpt2': 'GPT-2',
        'EleutherAI-pythia-160m': 'Pythia-160m'}

    dict_list = new_idx2dict_list[max_avg_key]
    csv_output_dir = Path("output/data_for_ultraviolet_and_light_bargraphs/")
    csv_output_dir.mkdir(exist_ok=True, parents=True)
    for mx, (model_name, dic) in tqdm(enumerate(zip(dynamic_models, dict_list))):
        assert dic['model_name'] == model_name
        uword = dic['uword']
        # remove prefix such as 'Ġ'
        if uword.startswith('Ġ'):
            uword = uword[1:]
        elif uword.startswith('##'):
            uword = uword[2:]
        norm_u = dic['norm_u']

        lword = dic['lword']
        if lword.startswith('Ġ'):
            lword = lword[1:]
        elif lword.startswith('##'):
            lword = lword[2:]
        norm_l = dic['norm_l']

        prod_word = f'{uword} $\odot$ {lword}'
        prod = norm_u * norm_l
        cos = dic['cos']
        logger.info(f'{model_name}: {uword} {lword} {cos:.3f}')

        labels = model2labels[model_name]
        ylabel = model2ylabel[model_name]
        if ica:
            ylabel += '\nNormalized IC Value'
        else:
            ylabel += '\nNormalized PC Value'
        top5_axis_idxs = np.argsort(-norm_u)[:5]
        top5_axis_idxs = sorted(top5_axis_idxs)

        normed_embed = model2normed_embed[model_name]
        id2word = modelid2word[model_name]
        data = []
        for color_idx, axis_idx in enumerate(top5_axis_idxs):
            row = {}
            row["Model"] = model2ylabel[model_name]
            row["Axis"] = axis_idx + 1
            top10_word_ids = np.argsort(-normed_embed[:, axis_idx])[:10]
            top10_words = [id2word[id] for id in top10_word_ids]
            show_words = []
            for word in top10_words:
                if word.startswith('Ġ'):
                    word = word[1:]
                elif word.startswith('##'):
                    word = word[2:]
                show_words.append(word)
                
            for topk, word in enumerate(show_words):
                row[f"Top{topk+1}"] = word
            data.append(row)

        df = pd.DataFrame(data)
        if not ica:
            csv_output_path = csv_output_dir /\
                f"pca_{model_name}_ultraviolet_top5.csv"
            logger.info(f"Saving to {csv_output_path}")
            df.to_csv(csv_output_path, index=False)

        for idx, (embed, word) in enumerate(zip(
            [norm_u, norm_l, prod],
            [uword, lword, prod_word]
        )):
            dim = len(embed)
            ax = axs[mx, idx]
            ax.bar(np.arange(len(embed)), embed,
                   color='black', alpha=0.25)

            for cdx, axis_idx in enumerate(top5_axis_idxs):
                dummy2 = np.zeros_like(embed)
                x = embed[axis_idx]
                dummy2[axis_idx] = x
                if ica:
                    label = labels[cdx]
                    label_with_axis_idx = f'{axis_idx + 1} [{label}]: {x:.3f}'
                else:
                    label = f'PC{axis_idx+1}'
                    label_with_axis_idx = f'[{label}]: {x:.3f}'

                ax.bar(np.arange(len(dummy2)), dummy2, color=colors[cdx],
                       label=label_with_axis_idx, width=1.5)

                if mx == 0:
                    ax.set_title(word, fontsize=title_fs, pad=15)

                # y range
                if model_name == 'EleutherAI-pythia-160m' or model_name == 'gpt2':
                    ax.set_ylim(-0.15, 0.7)
                    ax.set_yticks(np.arange(-0., 0.7, 0.2))
                else:
                    ax.set_ylim(-0.15, 0.35)
                    ax.set_yticks(np.arange(-0., 0.3, 0.2))

                # x label
                if mx == num_rows - 1:
                    ax.set_xlabel('Axis', fontsize=label_fs)
                # x ticks
                ax.set_xticks(np.arange(100, dim+1, 100))

                if idx == 0:
                    ax.set_ylabel(ylabel, fontsize=ylabel_fs)
                ax.legend(fontsize=legend_fs, loc=legend_loc)

                # ticks params
                ax.tick_params(labelsize=ts)

    output_dir = Path('output/camera_ready_images/')
    output_dir.mkdir(exist_ok=True, parents=True)
    if ica:
        output_path = output_dir / '4bargraph_ica.pdf'
    else:
        output_path = output_dir / '4bargraph_pca.pdf'

    # adjust
    plt.subplots_adjust(left=0.05, right=0.99,
                        top=0.96, bottom=0.06, wspace=0.1, hspace=0.3)
    logger.info(f'Saving to {output_path}')
    plt.savefig(output_path)


if __name__ == '__main__':
    main(ica=False)