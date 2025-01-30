import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from utils import pos_direct

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(ica):

    static_models = [
        "glove",
    ]
    logger.info(f"static_models: {static_models}")

    dynamic_models = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "EleutherAI-pythia-160m",
    ]
    logger.info(f"dynamic_models: {dynamic_models}")

    models = static_models + dynamic_models

    # Load embeddings
    input_dir = Path("output/embeddings/")

    model2dic_and_emb = {}

    # static
    for model_name in static_models:
        input_path = input_dir / f"{model_name}_dic_and_emb.pkl"
        with open(input_path, "rb") as f:
            word2id, id2word, _, pca_embed, ica_embed = pickle.load(f)
        model2dic_and_emb[model_name] = {
            "word2id": word2id,
            "id2word": id2word,
            "selected_embed": pos_direct(ica_embed) if ica else pca_embed,
        }

    # dynamic
    for model_name in dynamic_models:
        input_path = input_dir / f"{model_name}_dic_and_emb.pkl"

        with open(input_path, "rb") as f:
            word2id, id2word, _, pca_embed, ica_embed = pickle.load(f)
        model2dic_and_emb[model_name] = {
            "word2id": word2id,
            "id2word": id2word,
            "selected_embed": pos_direct(ica_embed) if ica else pca_embed,
        }

    ref_model_name = "glove"
    logger.info(f"referece model: {ref_model_name}")

    # registration
    other_model2dims = {}
    other_model2id2word = {}
    other_model2token2ids = defaultdict(dict)
    for model_name in model2dic_and_emb:
        if model_name == ref_model_name:
            ref_id2word = model2dic_and_emb[model_name]["id2word"]
            ref_embed = model2dic_and_emb[model_name]["selected_embed"]
            ref_dim = ref_embed.shape[1]
            continue

        other_model2dims[model_name] = model2dic_and_emb[model_name][
            "selected_embed"
        ].shape[1]
        other_id2word = model2dic_and_emb[model_name]["id2word"]
        other_model2id2word[model_name] = other_id2word
        for id_, token in enumerate(other_id2word):
            if model_name in dynamic_models:
                # remove count number
                token = "_".join(token.split("_")[:-1])

            # remove prefix such as '##'
            if token.startswith("##") and len(token) > 2:
                token = token[2:]
            elif token.startswith("Ġ") and len(token) > 1:
                token = token[1:]
            elif token.startswith("▁") and len(token) > 1:
                token = token[1:]
            if token not in other_model2token2ids[model_name]:
                other_model2token2ids[model_name][token] = []
            other_model2token2ids[model_name][token].append(id_)

    other_model_names = list(other_model2dims.keys())
    ref_axis_idx2corr_info = defaultdict(list)
    other_model2selected_corr_ref_axis_other_axis = {}
    for model_name in other_model_names:
        pairs = []
        for ref_id in range(len(ref_id2word)):
            ref_word = ref_id2word[ref_id]
            if ref_word not in other_model2token2ids[model_name]:
                continue
            other_ids = other_model2token2ids[model_name][ref_word]
            for other_id in other_ids:
                pairs.append((ref_id, other_id))

        ref_pair_embed = []
        other_pair_embed = []
        for ref_id, other_id in pairs:
            ref_pair_embed.append(ref_embed[ref_id])
            other_pair_embed.append(
                model2dic_and_emb[model_name]["selected_embed"][other_id]
            )
        ref_pair_embed = np.array(ref_pair_embed)
        other_pair_embed = np.array(other_pair_embed)
        logger.info(
            f"{ref_model_name}: {ref_pair_embed.shape}, "
            f"{model_name}: {other_pair_embed.shape}"
        )

        corr_ref_axis_other_axis = []
        logger.info("correlation calculation")
        for ref_axis_idx in tqdm(range(ref_dim)):
            for other_axis_idx in range(other_model2dims[model_name]):
                ref_axis = ref_pair_embed[:, ref_axis_idx]
                other_axis = other_pair_embed[:, other_axis_idx]
                corr = pearsonr(ref_axis, other_axis)[0]
                corr_ref_axis_other_axis.append((corr, ref_axis_idx, other_axis_idx))
        corr_ref_axis_other_axis.sort(reverse=True)

        # choose axis pairs greedily
        used_ref_axis = set()
        used_other_axis = set()
        selected_corr_ref_axis_other_axis = []
        for corr, ref_axis_idx, other_axis_idx in corr_ref_axis_other_axis:
            if ref_axis_idx in used_ref_axis or other_axis_idx in used_other_axis:
                continue
            ref_axis_idx2corr_info[ref_axis_idx].append(
                {
                    "model_name": model_name,
                    "other_axis_idx": other_axis_idx,
                    "corr": corr,
                }
            )
            selected_corr_ref_axis_other_axis.append(
                (corr, ref_axis_idx, other_axis_idx)
            )
            used_ref_axis.add(ref_axis_idx)
            used_other_axis.add(other_axis_idx)

        # sort by other_axis_idx for later index mapping
        selected_corr_ref_axis_other_axis.sort(key=lambda x: x[2])
        other_model2selected_corr_ref_axis_other_axis[model_name] = (
            selected_corr_ref_axis_other_axis
        )

    # we choose only the index
    # where all other models have the pair with the ref model
    valid_ref_axis_idxs = []
    for ref_axis_idx in sorted(ref_axis_idx2corr_info.keys()):
        corr_info_list = ref_axis_idx2corr_info[ref_axis_idx]
        if len(corr_info_list) == len(other_model_names):
            valid_ref_axis_idxs.append(ref_axis_idx)
    min_dim = len(valid_ref_axis_idxs)

    # index mapping
    ref_axis_idx2new_idx = {}
    for new_idx, ref_axis_idx in enumerate(valid_ref_axis_idxs):
        ref_axis_idx2new_idx[ref_axis_idx] = new_idx
    other_model2axis_idx2new_idx = defaultdict(dict)
    other_model2permutated_embed = {}
    for model_name in other_model_names:
        new_idx = 0
        for (
            _,
            ref_axis_idx,
            other_axis_idx,
        ) in other_model2selected_corr_ref_axis_other_axis[model_name]:
            if ref_axis_idx in ref_axis_idx2new_idx:
                other_model2axis_idx2new_idx[model_name][other_axis_idx] = new_idx
                new_idx += 1
        assert len(other_model2axis_idx2new_idx[model_name]) == min_dim

        # axis permutation
        W = np.zeros((min_dim, min_dim))
        for (
            _,
            ref_axis_idx,
            other_axis_idx,
        ) in other_model2selected_corr_ref_axis_other_axis[model_name]:
            if ref_axis_idx not in ref_axis_idx2new_idx:
                continue
            new_ref_axis_idx = ref_axis_idx2new_idx[ref_axis_idx]
            assert other_axis_idx in other_model2axis_idx2new_idx[model_name]
            new_other_axis_idx = other_model2axis_idx2new_idx[model_name][
                other_axis_idx
            ]
            W[new_other_axis_idx, new_ref_axis_idx] = 1

        # compress other model embeddings
        compress_other_embed = np.zeros_like(
            model2dic_and_emb[model_name]["selected_embed"][:, :min_dim]
        )

        other_embed = model2dic_and_emb[model_name]["selected_embed"]
        for other_axis_idx in range(other_model2dims[model_name]):
            if other_axis_idx not in other_model2axis_idx2new_idx[model_name]:
                continue
            new_other_axis_idx = other_model2axis_idx2new_idx[model_name][
                other_axis_idx
            ]
            compress_other_embed[:, new_other_axis_idx] = other_embed[:, other_axis_idx]

        # (N, min_dim) @ (min_dim, min_dim) = (N, min_dim)
        permutated_other_embed = compress_other_embed @ W
        other_model2permutated_embed[model_name] = permutated_other_embed

    # sort the axex of the ref model and other models
    # by the sum of the correlation with the ref model
    new_ref_axis_idx2sum_corr = {}
    for ref_axis_idx in range(ref_dim):
        if ref_axis_idx not in ref_axis_idx2new_idx:
            continue
        new_ref_axis_idx = ref_axis_idx2new_idx[ref_axis_idx]
        corr_info_list = ref_axis_idx2corr_info[ref_axis_idx]
        assert len(corr_info_list) == len(other_model_names)
        assert len(new_ref_axis_idx2sum_corr) == new_ref_axis_idx
        sum_corr = sum([corr_info["corr"] for corr_info in corr_info_list])
        new_ref_axis_idx2sum_corr[new_ref_axis_idx] = sum_corr

    sorted_new_ref_aixs_idxs = sorted(
        new_ref_axis_idx2sum_corr.keys(),
        key=lambda x: new_ref_axis_idx2sum_corr[x],
        reverse=True,
    )
    assert len(sorted_new_ref_aixs_idxs) == min_dim
    sorted_ref_embed = np.zeros_like(ref_embed[:, :min_dim])
    other_model2sorted_other_embed = {}
    for model_name in other_model_names:
        other_model2sorted_other_embed[model_name] = np.zeros_like(
            other_model2permutated_embed[model_name][:, :min_dim]
        )
    for axis_idx, new_ref_axis_idx in enumerate(sorted_new_ref_aixs_idxs):
        sorted_ref_embed[:, axis_idx] = ref_embed[:, new_ref_axis_idx]
        for model_name in other_model_names:
            other_model2sorted_other_embed[model_name][:, axis_idx] = (
                other_model2permutated_embed[model_name][:, new_ref_axis_idx]
            )

    # save the sorted embeddings
    output_dir = Path("output/axis_matching")
    output_dir.mkdir(exist_ok=True, parents=True)
    file_name = "sorted_axis_" + ("ica_" if ica else "pca_") + "_".join(models) + ".pkl"
    output_path = output_dir / file_name
    logger.info(f"save to {output_path}")
    data = {
        "ref_model_name": ref_model_name,
        "other_model_names": other_model_names,
        "ref_id2word": ref_id2word,
        "other_model2id2word": other_model2id2word,
        "other_model2token2ids": other_model2token2ids,
        "sorted_ref_embed": sorted_ref_embed,
        "other_model2sorted_other_embed": other_model2sorted_other_embed,
    }
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main(ica=True)
    main(ica=False)
