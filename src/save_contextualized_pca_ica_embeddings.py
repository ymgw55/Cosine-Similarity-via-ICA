import argparse
import logging
import pickle
import pickle as pkl
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA, FastICA

from utils import get_model_names

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Save PCA and ICA embeddings.")

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-10)

    return parser.parse_args()


def main():

    args = parse_args()
    logger.info(args)
    model_name = args.model_name

    encoder_model_names, decoder_model_names = get_model_names()
    model_names = encoder_model_names + decoder_model_names
    assert model_name in model_names

    output_dir = Path("output/embeddings")
    input_path = output_dir / f'{model_name.replace("/", "-")}.pkl'

    assert input_path.exists()
    logger.info(f"input_path: {input_path}")
    with open(input_path, "rb") as f:
        token_list, embedding_list = pickle.load(f)
    id2word = np.array(token_list)
    word2id = {word: i for i, word in enumerate(id2word)}
    embed = np.array(embedding_list).astype(np.float64)

    logger.info(f"id2word example: {id2word[:10]}")
    logger.info(f"embed.shape: {embed.shape}")
    rng = np.random.RandomState(args.seed)

    # centering
    embed_ = embed - embed.mean(axis=0)

    # PCA
    pca_params = {"random_state": rng}
    logger.info(f"pca_params: {pca_params}")
    pca = PCA(random_state=rng)
    pca_embed = pca.fit_transform(embed_)
    pca_embed = pca_embed / pca_embed.std(axis=0)

    # ICA
    ica_params = {
        "n_components": None,
        "random_state": rng,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "whiten": False,
    }
    logger.info(f"ica_params: {ica_params}")
    ica = FastICA(**ica_params)
    ica.fit(pca_embed)
    R = ica.components_.T
    ica_embed = pca_embed @ R

    # save embeddings
    dic_and_emb = (word2id, id2word, embed, pca_embed, ica_embed)
    emb_path = output_dir / f'{model_name.replace("/", "-")}_dic_and_emb.pkl'
    logger.info(f"emb_path: {emb_path}")
    with open(emb_path, "wb") as f:
        pkl.dump(dic_and_emb, f)


if __name__ == "__main__":
    main()
