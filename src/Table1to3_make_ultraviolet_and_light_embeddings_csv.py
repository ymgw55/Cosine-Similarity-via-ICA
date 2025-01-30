import logging
import pickle as pkl
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import pos_direct

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(ica):

    dynamic_models = [
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
            word2id, id2word, _, pca_embed, ica_embed = pkl.load(f)
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

    dict_list = new_idx2dict_list[max_avg_key]
    model_lists = ["gpt2", "EleutherAI-pythia-160m"]

    data = dict()
    for mx, model_name in enumerate(model_lists):
        for dic in dict_list:
            mdl = dic["model_name"]
            if model_name == mdl:
                break
        assert dic["model_name"] == model_name

        row = dict()

        uword = dic["uword"]
        # remove prefix such as 'Ġ'
        if uword.startswith("Ġ"):
            uword = uword[1:]
        elif uword.startswith("##"):
            uword = uword[2:]
        norm_u = dic["norm_u"]
        row["uword"] = uword
        row["norm_u"] = norm_u

        lword = dic["lword"]
        if lword.startswith("Ġ"):
            lword = lword[1:]
        elif lword.startswith("##"):
            lword = lword[2:]
        norm_l = dic["norm_l"]
        row["lword"] = lword
        row["norm_l"] = norm_l

        prod = norm_u * norm_l
        row["pword"] = f"{uword}_{lword}"
        row["prod"] = prod

        for vec_word, vec in [("uword", norm_u), ("lword", norm_l), ("pword", prod)]:
            top5_axis_idxs = np.argsort(-vec)[:5]
            top5_axis_idxs = sorted(top5_axis_idxs)
            row[f"{vec_word}_top5_axis_idxs"] = top5_axis_idxs

            normed_embed = model2normed_embed[model_name]
            id2word = modelid2word[model_name]

            top10_words_list = []
            for color_idx, axis_idx in enumerate(top5_axis_idxs):
                top10_word_ids = np.argsort(-normed_embed[:, axis_idx])[:10]
                top10_words = [id2word[id_] for id_ in top10_word_ids]
                top10_words = [
                    word[1:] if word.startswith("Ġ") else word for word in top10_words
                ]
                top10_words_list.append(top10_words)
            row[f"{vec_word}_top10_words"] = top10_words_list
        data[model_name] = row

    # glove
    emb_path = "output/embeddings/glove_dic_and_emb.pkl"

    with open(emb_path, "rb") as f:
        word2id, id2word, embed, pca_embed, ica_embed = pkl.load(f)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]

    ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)
    pca_embed = pca_embed / np.linalg.norm(pca_embed, axis=1, keepdims=True)

    if ica:
        embed = ica_embed
    else:
        embed = pca_embed

    uv_embed = embed[word2id["ultraviolet"]]
    light_embed = embed[word2id["light"]]
    prod = uv_embed * light_embed

    row = dict()
    row["uword"] = "ultraviolet"
    row["norm_u"] = uv_embed
    row["lword"] = "light"
    row["norm_l"] = light_embed
    row["pword"] = "ultraviolet_light"
    row["prod"] = prod

    uv_top5_axis_idxs = np.argsort(-uv_embed)[:5]
    uv_top5_axis_idxs = sorted(uv_top5_axis_idxs)
    uv_top10_word_list = []
    for idx in uv_top5_axis_idxs:
        top10_word_idx = np.argsort(-embed[:, idx])[:10]
        words = [id2word[i] for i in top10_word_idx]
        uv_top10_word_list.append(words)
    row["uword_top5_axis_idxs"] = uv_top5_axis_idxs
    row["uword_top10_words"] = uv_top10_word_list

    light_top5_axis_idxs = np.argsort(-light_embed)[:5]
    light_top5_axis_idxs = sorted(light_top5_axis_idxs)
    light_top10_word_list = []
    for idx in light_top5_axis_idxs:
        top10_word_idx = np.argsort(-embed[:, idx])[:10]
        words = [id2word[i] for i in top10_word_idx]
        light_top10_word_list.append(words)
    row["lword_top5_axis_idxs"] = light_top5_axis_idxs
    row["lword_top10_words"] = light_top10_word_list

    prod_top5_axis_idxs = np.argsort(-prod)[:5]
    prod_top5_axis_idxs = sorted(prod_top5_axis_idxs)
    prod_top10_word_list = []
    for idx in prod_top5_axis_idxs:
        top10_word_idx = np.argsort(-embed[:, idx])[:10]
        words = [id2word[i] for i in top10_word_idx]
        prod_top10_word_list.append(words)
    row["pword_top5_axis_idxs"] = prod_top5_axis_idxs
    row["pword_top10_words"] = prod_top10_word_list

    data["glove"] = row

    output_dir = Path("output/ultraviolet_and_light")
    output_dir.mkdir(exist_ok=True, parents=True)
    if ica:
        output_path = output_dir / "ica_data.pkl"
    else:
        output_path = output_dir / "pca_data.pkl"
    with open(output_path, "wb") as f:
        pkl.dump(data, f)

    # save as csv
    csv_dir = Path("src/mathematica/data")
    csv_dir.mkdir(exist_ok=True, parents=True)
    for model_name, row in data.items():
        csv_data = []
        norm_u = row["norm_u"]
        norm_l = row["norm_l"]
        prod = row["prod"]

        # save 3 embeddings as csv
        csv_data = []
        for vec_word, vec in [
            ("ultraviolet", norm_u),
            ("light", norm_l),
        ]:
            emb_row = []
            emb_row.append(vec_word)
            emb_row += vec.tolist()
            csv_data.append(emb_row)

        # csv_data is 2 x (1 + dim)
        df = pd.DataFrame(csv_data)

        if ica:
            csv_path = csv_dir / f"{model_name}_ica.csv"
        else:
            csv_path = csv_dir / f"{model_name}_pca.csv"

        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, header=False)


if __name__ == "__main__":
    main(ica=True)
    main(ica=False)
