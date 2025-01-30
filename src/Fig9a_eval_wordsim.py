import logging
import pickle as pkl
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from web.datasets.similarity import (fetch_MEN, fetch_MTurk, fetch_RG65,
                                     fetch_RW, fetch_SimLex999, fetch_WS353)
from web.embedding import Embedding
from web.evaluate import evaluate_similarity
from web.vocabulary import OrderedVocabulary

from utils import pos_direct

warnings.filterwarnings('ignore')

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class WordsimEmbedding(Embedding):
    # override
    def __init__(self, vocab, vectors, p):
        super().__init__(vocab, vectors)
        self.p = p

    @staticmethod
    def from_words_and_vectors(words, vectors, p):
        vocab = OrderedVocabulary(words)
        return WordsimEmbedding(vocab, vectors, p)


def main():

    # seed
    np.random.seed(0)

    similarity_tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "WS353R": fetch_WS353(which="relatedness"),
        "WS353S": fetch_WS353(which="similarity"),
        "SimLex999": fetch_SimLex999(),
        "RW": fetch_RW(),
        "RG65": fetch_RG65(),
        "MTurk": fetch_MTurk(),
    }

    emb_path = "output/embeddings/glove_dic_and_emb.pkl"
    with open(emb_path, 'rb') as f:
        word2id, id2word, embed, pca_embed, ica_embed = pkl.load(f)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]

    data = []
    ps = [1, 2, 5, 10, 20, 50, 100, 200, 300]
    for p in ps:
        for emb_type in ['pca', 'ica']:
            if emb_type == 'pca':
                embed = pca_embed
            elif emb_type == 'ica':
                embed = ica_embed

            w = WordsimEmbedding.from_words_and_vectors(
                id2word, embed, p)

            # sim tasks
            for task_name, task in similarity_tasks.items():
                spearman = evaluate_similarity(w, task.X, task.y)
                if np.isnan(spearman):
                    spearman = 0
                row = {
                    'emb_type': emb_type,
                    'p': p,
                    'task_type': 'similarity',
                    'task': task_name,
                    'spearman': spearman,
                }
                logger.info(row)
                data.append(row)

    # save
    df = pd.DataFrame(data)
    output_dir = Path("output/evaluation")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / 'wordsim.csv'
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()