## Computing $p$-values using Mathematica

### Preliminary

As described in [../../README.md](../../README.md), you must first run [../Table1to3_make_ultraviolet_and_light_embeddings_csv.py](../Table1to3_make_ultraviolet_and_light_embeddings_csv.py).

This script outputs the component values (in CSV format) of the normalized embeddings of "ultraviolet", "light", and "ultraviolet $\odot$ light" for GloVe, GPT-2, and Pythia-160m.
As an example, the output used in the paper is placed under [data](data).

### Run

[product_of_two_normal.nb](product_of_two_normal.nb) is a Mathematica notebook for computing $p$-values for the component values of these vectors. In particular, it shows the computation results for GloVe. To do the same for GPT-2 or Pythia-160m, simply change the path to the input CSV file. A PDF rendering of the notebook is provided in [product_of_two_normal.pdf](product_of_two_normal.pdf).

The experimental results for GloVe, GPT-2, and Pythia-160m are available in [../../data/ultraviolet_and_light_pvalues](../../data/ultraviolet_and_light_pvalues).