import pickle as pkl

import numpy as np
import pandas as pd

from utils import pos_direct


def main():

    emb_path = "output/embeddings/glove_dic_and_emb.pkl"
    with open(emb_path, "rb") as f:
        word2id, id2word, _, _, ica_embed = pkl.load(f)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]
    ica_embed = ica_embed / np.linalg.norm(ica_embed, axis=1, keepdims=True)

    src_word = "ultraviolet"


    top5_idx = np.argsort(-ica_embed[word2id[src_word]])[:5]
    top5_idx = sorted(top5_idx)
    # we checked in advance that the meanings of the top 5 axes are as follows:
    labels = ("chemistry", "biology", "space", "spectrum", "virology")
    if labels == ("chemistry", "biology", "space", "spectrum", "virology"):
        # if you conduct the experiment with a different setting, 
        # you may need to change the labels accordingly.
        assert tuple([i+1 for i in top5_idx]) == (53, 68, 141, 194, 197)
    labels_with_axis_index = [f"{top5_idx[i]+1} [{labels[i]}]" for i in range(5)]

    src_idx = word2id[src_word]
    src_emb = ica_embed[src_idx]
    norm_src_emb = src_emb / np.linalg.norm(src_emb)

    rows = []
    for axis_idx in top5_idx:
        tgt_word = id2word[np.argsort(-ica_embed[:, axis_idx])[0]]
        tgt_idx = word2id[tgt_word]
        tgt_emb = ica_embed[tgt_idx]
        norm_tgt_emb = tgt_emb / np.linalg.norm(tgt_emb)

        prod_vec = norm_src_emb * norm_tgt_emb
        cos_sim = np.sum(prod_vec)

        row = dict()
        row["word"] = tgt_word
        for axis_jdx, col in zip(top5_idx, labels_with_axis_index):
            prod = prod_vec[axis_jdx]
            row[col] = f"{prod:.3f}"

        row[f"cossim with {src_word}"] = f"{cos_sim:.3f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df)


if __name__ == "__main__":
    main()
