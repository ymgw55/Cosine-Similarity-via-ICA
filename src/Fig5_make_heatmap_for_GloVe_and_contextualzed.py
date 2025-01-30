import logging
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(ica=True):

    if ica:
        ica_str = "ica"
    else:
        ica_str = "pca"

    # Load the data
    input_path = f"output/axis_matching/sorted_axis_{ica_str}_glove_bert-base-uncased_roberta-base_gpt2_EleutherAI-pythia-160m.pkl"  # noqa

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    ref_model_name = data["ref_model_name"]
    other_model_names = data["other_model_names"]
    ref_id2word = data["ref_id2word"]
    other_model2id2word = data["other_model2id2word"]
    other_model2token2ids = data["other_model2token2ids"]
    sorted_ref_embed = data["sorted_ref_embed"]
    other_model2sorted_other_embed = data["other_model2sorted_other_embed"]

    logger.info(f"ref_model_name: {ref_model_name}")
    logger.info(f"other_model_names: {other_model_names}")

    normed_sorted_ref_embed = sorted_ref_embed / np.linalg.norm(
        sorted_ref_embed, axis=1, keepdims=True
    )

    normed_other_model2sorted_other_embed = {
        model_name: embed / np.linalg.norm(embed, axis=1, keepdims=True)
        for model_name, embed in other_model2sorted_other_embed.items()
    }

    axis_num = 100
    max_top_word_num = 5
    top_ref_words_list = []
    for axis_idx in range(axis_num):
        ref_axis = normed_sorted_ref_embed[:, axis_idx]
        ref_word_idxs = np.argsort(-ref_axis)

        top_word_num = 0
        ref_id2model2ids = {}
        for ref_word_idx in ref_word_idxs:
            ref_word = ref_id2word[ref_word_idx]
            exist_flag = True
            model2ids = {}
            for model_name in other_model_names:
                token2ids = other_model2token2ids[model_name]
                if ref_word not in token2ids:
                    exist_flag = False
                    break
                model2ids[model_name] = token2ids[ref_word]

            if not exist_flag:
                continue

            top_word_num += 1
            ref_id2model2ids[ref_word_idx] = model2ids
            if top_word_num >= max_top_word_num:
                break
        top_ref_words_list.append(ref_id2model2ids)

    model2embeds = defaultdict(list)  # (max_top_word_num * axis_num, axis_num)
    model2words = defaultdict(list)
    for axis_idx, ref_id2model2ids in enumerate(top_ref_words_list):
        assert len(ref_id2model2ids) == max_top_word_num
        for ref_word_idx, model2ids in ref_id2model2ids.items():
            ref_embed = normed_sorted_ref_embed[ref_word_idx]
            model2embeds[ref_model_name].append(ref_embed[:axis_num])
            ref_word = ref_id2word[ref_word_idx]
            model2words[ref_model_name].append(ref_word)

            for model_name, ids in model2ids.items():
                other_id2word = other_model2id2word[model_name]
                # sort word idx by the element of the axis
                id_and_x = []
                for other_word_idx in ids:
                    other_embed = normed_other_model2sorted_other_embed[model_name][
                        other_word_idx
                    ]
                    x = other_embed[axis_idx]
                    id_and_x.append((other_word_idx, x))
                id_and_x = sorted(id_and_x, key=lambda x: x[1], reverse=True)
                for other_word_idx, _ in id_and_x[:1]:
                    other_embed = normed_other_model2sorted_other_embed[model_name][
                        other_word_idx
                    ]
                    model2embeds[model_name].append(other_embed[:axis_num])
                    other_word = other_id2word[other_word_idx]
                    model2words[model_name].append(other_word)

    for model_name, embeds in model2embeds.items():
        model2embeds[model_name] = np.array(embeds)
        words = model2words[model_name]

        # remove prefix such as 'Ġ'
        clean_words = []
        for word in words:
            if word.startswith("Ġ"):
                clean_words.append(word[1:])
            elif word.startswith("##"):
                clean_words.append(word[2:])
            else:
                clean_words.append(word)
        model2words[model_name] = clean_words

        logger.info(f"{model_name}, {model2embeds[model_name].shape}, {words[:25]}")

    # Plot embedding heatmap

    def model_name2title(model_name):
        if model_name == "glove":
            return "GloVe"
        elif model_name == "fasttext":
            return "fastText"
        elif model_name == "bert-base-uncased":
            return "BERT-base"
        elif model_name == "roberta-base":
            return "RoBERTa-base"
        elif model_name == "gpt2":
            return "GPT-2"
        elif model_name == "EleutherAI-pythia-160m":
            return "Pythia-160m"

    col = 1 + len(other_model_names)
    figx = 5 * col
    figy = 4 * 2
    wspace = 0.5
    fig = plt.figure(figsize=(figx, figy))
    gs1 = gridspec.GridSpec(
        1,
        col,
        figure=fig,
        width_ratios=[1] * col,
        wspace=wspace,
        bottom=0.635,
        top=0.92,
    )
    gs2 = gridspec.GridSpec(
        1, col, figure=fig, width_ratios=[1] * col, wspace=wspace, bottom=0.05, top=0.56
    )
    fig.subplots_adjust(left=0.055, right=0.93)
    gss = [gs1, gs2]
    cb_ax = fig.add_axes([0.95, 0.05, 0.015, 0.87])
    cb_ax.tick_params(labelsize=30)
    zoom_axis_num = 5
    zoom_word_num = zoom_axis_num * max_top_word_num

    def sub_figure(embeds, words, ax, title, cb_ax, zoom):
        if zoom:
            show_title = False
            show_words = True
        else:
            show_title = True
            show_words = False

        # hyperparameters
        ls = 25
        title_fs = 35
        s = 1 if show_words else 0.05
        padding = 0.001
        lw = 5 if show_words else 1
        n = len(embeds)
        dim = len(embeds[0])
        tick_fs = 14

        cbar = cb_ax is not None
        g = sns.heatmap(
            embeds,
            yticklabels=words,
            cmap="magma_r",
            ax=ax,
            vmin=-0.04,
            vmax=1.0,
            cbar_ax=cb_ax,
            cbar=cbar,
        )
        g.tick_params(left=False, bottom=True, labelsize=ls)

        if show_title:
            ax.text(
                0.5,
                1.125,
                title,
                fontsize=title_fs,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

        if show_words:
            yticklabels = ax.get_yticklabels()
            ax.set_yticklabels(yticklabels, rotation=0, fontsize=tick_fs)
            ax.set_xlim(-dim * padding, dim * (s + 3 * padding))
            ax.set_ylim(n * (s + 3 * padding), -n * 3 * padding)
            ax.set_xticks(np.arange(0.5, 5, 1))
            ax.set_xticklabels(range(1, 6), rotation=0)
        else:
            ax.set_yticklabels([])
            # no rotation
            ax.set_xticks(range(20, dim - 1, 20))
            ax.set_xticklabels(range(20, dim - 1, 20), rotation=0)
            ax.set_xlim(-dim * 2 * padding, dim)
            ax.set_ylim(n, -n * 2 * padding)

        # coordinate of the rectangle
        x = -padding * dim
        y = -padding * n
        width = dim * (s + padding)
        height = n * (s + padding)

        # Draw the border of the rectangle
        lines = [
            Line2D([x, x + width], [y, y], lw=lw, color="black"),
            Line2D([x + width, x + width], [y, y + height], lw=lw, color="black"),
            Line2D([x + width, x], [y + height, y + height], lw=lw, color="black"),
            Line2D([x, x], [y + height, y], lw=lw, color="black"),
        ]
        for line in lines:
            ax.add_line(line)

    for fig_idx in range(2):
        zoom = bool(fig_idx)

        if zoom:
            model2embeds_zoom = {}
            model2words_zoom = {}
            for model_name, embeds in model2embeds.items():
                model2embeds_zoom[model_name] = embeds[:zoom_word_num, :zoom_axis_num]
                model2words_zoom[model_name] = model2words[model_name][:zoom_word_num]
            cb_ax = None

        for sub_idx, model_name in enumerate([ref_model_name] + other_model_names):
            title = model_name2title(model_name)
            if zoom:
                embeds = model2embeds_zoom[model_name]
                words = model2words_zoom[model_name]
            else:
                embeds = model2embeds[model_name]
                words = model2words[model_name]
            ax = fig.add_subplot(gss[fig_idx][sub_idx])
            logger.info(f"embeds.shape: {embeds.shape}")

            sub_figure(embeds, words, ax, title, cb_ax, zoom)

    # save
    output_dir = Path("output/camera_ready_images")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = output_dir / (f"heatmap_for_GloVe_and_contextualized_{ica_str}.png")
    logger.info(f"Save the heatmap to {output_path}")
    plt.savefig(output_path, dpi=200)


if __name__ == "__main__":
    main(ica=True)
    main(ica=False)
