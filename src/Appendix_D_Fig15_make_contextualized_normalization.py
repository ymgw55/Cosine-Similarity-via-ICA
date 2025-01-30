import logging
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from PIL import Image
from tqdm import tqdm

from utils import pos_direct

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    dynamic_models = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "EleutherAI-pythia-160m",
    ]

    model2emb_list = {}

    input_dir = Path("output/embeddings/")
    for model_name in dynamic_models:
        logger.info(model_name)

        input_path = input_dir / f"{model_name}_dic_and_emb.pkl"

        with open(input_path, "rb") as f:
            word2id, id2word, _, pca_embed, ica_embed = pickle.load(f)

        # ica
        ica_embed = pos_direct(ica_embed)
        skewness = np.mean(ica_embed**3, axis=0)
        skew_sort_idx = np.argsort(-skewness)
        ica_embed = ica_embed[:, skew_sort_idx]
        ica_norms = np.linalg.norm(ica_embed, axis=1)
        normed_ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)

        # pca
        pca_norms = np.linalg.norm(pca_embed, axis=1)
        normed_pca_embed = pca_embed / np.linalg.norm(pca_embed, axis=1, keepdims=True)

        model2emb_list[model_name] = [
            [ica_embed, normed_ica_embed, ica_norms, "ica"],
            [pca_embed, normed_pca_embed, pca_norms, "pca"],
        ]

    def sci_notation(tick_val, pos):
        if tick_val != 0:
            exponent = int(np.log10(tick_val))
            base = tick_val / 10**exponent
            return f"{base:.0f}e{exponent}"
        else:
            return "0"

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

    dim = 149

    model_xca_state2save_path = {}
    output_dir = Path("output/camera_ready_images")
    output_dir.mkdir(exist_ok=True, parents=True)
    for model_name, emb_list in model2emb_list.items():
        for idx, (embed, normed_embed, norms, xca) in enumerate(emb_list):

            def func(embed, normed_embed, norms, xca, dim, model_name):
                sorted_embed = np.sort(embed, axis=1)
                n, emb_dim = embed.shape
                sorted_normed_embed = np.sort(normed_embed, axis=1)

                ls = 30
                ts = 20

                unnormed_idxs = np.argsort(-embed[:, dim])
                unnormed_i2r = {}
                for r, i in enumerate(unnormed_idxs):
                    unnormed_i2r[i] = r + 1

                unnormed_i2inner_r = {}
                for i in tqdm(range(n)):
                    x = embed[i, dim]

                    r = emb_dim - 1 - np.searchsorted(sorted_embed[i], x)
                    assert 0 <= r < emb_dim

                    unnormed_i2inner_r[i] = r + 1

                normed_idxs = np.argsort(-normed_embed[:, dim])
                normed_i2r = {}
                for r, i in enumerate(normed_idxs):
                    normed_i2r[i] = r + 1

                normed_i2inner_r = {}
                for i in tqdm(range(n)):
                    x = normed_embed[i, dim]

                    r = emb_dim - 1 - np.searchsorted(sorted_normed_embed[i], x)
                    assert 0 <= r < emb_dim

                    normed_i2inner_r[i] = r + 1

                for i2r, i2inner_r, state in [
                    (unnormed_i2r, unnormed_i2inner_r, "before"),
                    (normed_i2r, normed_i2inner_r, "after"),
                ]:
                    xs = []
                    ys = []
                    cs = []
                    for i in range(n):
                        xs.append(i2r[i])
                        ys.append(i2inner_r[i])
                        cs.append(norms[i])
                    xs = np.array(xs)
                    ys = np.array(ys)
                    cs = np.array(cs)

                    fig, ax = plt.subplots(figsize=(8, 7))

                    # adjust ax
                    plt.subplots_adjust(left=0.17, right=0.93, top=0.9, bottom=0.15)

                    # color by nor
                    title = model_name2title(model_name)
                    ax.set_title(f"{title} {xca.upper()[:-1]}{dim + 1}", fontsize=ls)
                    ax.set_xlabel("Rank along embeddings", fontsize=ls)
                    ax.set_ylabel("Rank along axes", fontsize=ls)

                    # ticks
                    ax.set_xticks(range(10000, n + 1, 10000))
                    ax.set_yticks(range(200, emb_dim + 1, 200))

                    # tick size
                    ax.tick_params(axis="x", which="major", labelsize=ts)
                    ax.tick_params(axis="y", which="major", labelsize=ts)

                    sorted_by_cs = np.argsort(cs)
                    # seismic
                    scatter = ax.scatter(
                        xs[sorted_by_cs],
                        ys[sorted_by_cs],
                        c=cs[sorted_by_cs],
                        cmap="seismic",
                        s=1,
                    )

                    img_path = (
                        output_dir / f"{model_name}_{state}_normalization_{xca}.png"
                    )
                    model_xca_state2save_path[(model_name, xca, state)] = img_path

                    # add color bar and label, with fontsize
                    cbar = plt.colorbar(scatter)
                    cbar.set_label("Norm", fontsize=ls, rotation=270, labelpad=30)
                    cbar.ax.tick_params(labelsize=ts)

                    plt.gca().xaxis.set_major_formatter(FuncFormatter(sci_notation))
                    logger.info(f"Saving to {img_path}")
                    plt.savefig(img_path, dpi=150)
                    plt.close()

            func(embed, normed_embed, norms, xca, dim, model_name)

    def save_concat_to_2x1(image_paths, output_path):
        images = [Image.open(image) for image in image_paths]
        widths, heights = zip(*(i.size for i in images))
        total_width = max(widths)
        total_height = sum(heights)
        new_image = Image.new("RGB", (total_width, total_height))

        y_offset = 0
        for img in images:
            new_image.paste(img, (0, y_offset))
            y_offset += img.height

        logger.info(f"Saving to {output_path}")
        new_image.save(output_path, dpi=(150, 150))

    def save_concat_to_1x4(image_paths, output_path):
        images = [Image.open(image) for image in image_paths]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        total_height = max(heights)
        new_image = Image.new("RGB", (total_width, total_height))

        x_offset = 0
        for img in images:
            new_image.paste(img, (x_offset, 0))
            x_offset += img.width

        logger.info(f"Saving to {output_path}")
        new_image.save(output_path, dpi=(150, 150))

    state2save_path = defaultdict(list)
    for model_name in dynamic_models:
        for state in ["before", "after"]:
            image_paths = []
            for xca in ["ica", "pca"]:
                image_paths.append(model_xca_state2save_path[(model_name, xca, state)])

            output_path = output_dir / f"{model_name}_{state}_normalization.png"
            save_concat_to_2x1(image_paths, output_path)
            state2save_path[state].append(output_path)

    for state in ["before", "after"]:
        image_paths = state2save_path[state]
        assert len(image_paths) == 4
        output_path = output_dir / f"contextualized_{state}_normalization.png"
        save_concat_to_1x4(image_paths, output_path)


if __name__ == "__main__":
    main()
