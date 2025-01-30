import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd


def main():

    models = ["glove", "gpt2", "EleutherAI-pythia-160m"]
    output_dir = Path('output/ultraviolet_and_light_pvalues')
    output_dir.mkdir(exist_ok=True, parents=True)
    for model in models:
        rows = []
        for ica in [True, False]:
            input_dir = Path("output/ultraviolet_and_light")
            if ica:
                input_path = input_dir / "ica_data.pkl"
            else:
                input_path = input_dir / "pca_data.pkl"
            with open(input_path, "rb") as f:
                data = pkl.load(f)

            model_dict = data[model]
            uv_embed = model_dict["norm_u"]
            uv_top5_axis_idxs = model_dict["uword_top5_axis_idxs"]
            uv_top10_word_list = model_dict["uword_top10_words"]

            light_embed = model_dict["norm_l"]
            light_top5_axis_idxs = model_dict["lword_top5_axis_idxs"]
            light_top10_word_list = model_dict["lword_top10_words"]

            prod = uv_embed * light_embed
            prod_top10_word_list = model_dict["pword_top10_words"]
            prod_top5_axis_idxs = model_dict["pword_top5_axis_idxs"]

            dim = len(uv_embed)

            # load pvalues
            pvalues_dir = Path("data/ultraviolet_and_light_pvalues")
            pvalues_path = pvalues_dir / f"{model}_{'ica' if ica else 'pca'}-pvalue.csv"
            df = pd.read_csv(pvalues_path, header=None).astype(np.float64)
            # df to numpy
            pvalues = df.to_numpy().astype(np.float64)
            assert pvalues.shape == (3, dim)
            uv_pvalues = pvalues[0]
            light_pvalues = pvalues[1]
            prod_pvalues = pvalues[2]

            # show pvalues
            words = ["ultraviolet", "light", "ultraviolet_light"]
            for i, (word, vec, top5_axis_idxs, top10_word_list, pvalue) in enumerate(
                zip(
                    words,
                    [uv_embed, light_embed, prod],
                    [uv_top5_axis_idxs, light_top5_axis_idxs, prod_top5_axis_idxs],
                    [uv_top10_word_list, light_top10_word_list, prod_top10_word_list],
                    [uv_pvalues, light_pvalues, prod_pvalues],
                )
            ):
                axis_idx2words = dict()
                for axis_idx, words in zip(top5_axis_idxs, top10_word_list):
                    axis_idx2words[axis_idx] = words

                sorted_top5_axis_idxs = sorted(top5_axis_idxs, key=lambda x: -vec[x])
                for axis_idx in sorted_top5_axis_idxs:

                    row = {
                        'transform': 'ica' if ica else 'pca',
                        'word': word,
                        'axis': axis_idx + 1,
                        'top10_words': axis_idx2words[axis_idx],
                        'value': f"{vec[axis_idx]:.3f}",
                        'pvalue': f"{pvalue[axis_idx]:.2e}",
                        'Bonferroni': f"{min(1, pvalue[axis_idx]*dim):.2e}",
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)
        output_path = output_dir / f"{model}.csv"
        df.to_csv(output_path, index=False)
        

if __name__ == "__main__":
    main()
