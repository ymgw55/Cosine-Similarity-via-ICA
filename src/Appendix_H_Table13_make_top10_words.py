import logging
import pickle as pkl
from pathlib import Path

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
    emb_path = "output/embeddings/glove_dic_and_emb.pkl"
    with open(emb_path, "rb") as f:
        word2id, id2word, embed, pca_embed, ica_embed = pkl.load(f)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]
    ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)

    src_word = "woman"

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

    org_cand_words, org_sims = nearest_words(src_emb, topk=100)
    org_word2sim = {word: sim for word, sim in zip(org_cand_words, org_sims)}

    ablating_emb = src_emb.copy()
    ablating_emb = ablating_emb / np.linalg.norm(ablating_emb)
    ablating_emb[axis_idx] = 0

    ablating_cand_words, sims = nearest_words(ablating_emb, topk=100)
    ablating_word2sim = {word: sim for word, sim in zip(ablating_cand_words, sims)}

    rows = []
    # exclude the query word itself from the list
    for hit_word in org_cand_words[1:11]:
        sim_with_woman = f"{org_word2sim.get(hit_word, 'None'):.3f}"
        sim_with_woman_minus_female = f"{ablating_word2sim.get(hit_word, 'None'):.3f}"
        diff = f"{float(sim_with_woman) - float(sim_with_woman_minus_female):.3f}"

        row = {
            "query_word": src_word,
            "hit_word": hit_word,
            "sim_with_woman": sim_with_woman,
            "sim_with_woman_minus_female": sim_with_woman_minus_female,
            "diff": diff,
        }
        logger.info(row)
        rows.append(row)

    # exclude the query word itself from the list
    for hit_word in ablating_cand_words[1:11]:
        sim_with_woman = f"{org_word2sim.get(hit_word, 'None'):.3f}"
        sim_with_woman_minus_female = f"{ablating_word2sim.get(hit_word, 'None'):.3f}"
        diff = f"{float(sim_with_woman) - float(sim_with_woman_minus_female):.3f}"
        row = {
            "query_word": f"{src_word}_minus_female",
            "hit_word": hit_word,
            "sim_with_woman": sim_with_woman,
            "sim_with_woman_minus_female": sim_with_woman_minus_female,
            "diff": diff,
        }
        logger.info(row)
        rows.append(row)

    df = pd.DataFrame(rows)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "Appendix_H_Table13.csv"
    logger.info(f"Save the result to {output_path}")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
