import argparse
import logging
import pickle as pkl
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from web.datasets.analogy import fetch_google_analogy, fetch_msr_analogy
from web.embedding import Embedding
from web.evaluate import evaluate_ps_analogy
from web.vocabulary import OrderedVocabulary

from utils import pos_direct

warnings.filterwarnings('ignore')

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AnalogyEmbedding(Embedding):
    # override
    def __init__(self, vocab, vectors, ps):
        super().__init__(vocab, vectors)
        self.ps = ps

    @staticmethod
    def from_words_and_vectors(words, vectors, ps):
        vocab = OrderedVocabulary(words)
        return AnalogyEmbedding(vocab, vectors, ps)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate analogy tasks.")

    parser.add_argument("--emb_type", type=str, default="ica")
    parser.add_argument("--task_name", type=str, default="Google")

    return parser.parse_args()


def main():

    args = parse_args()
    logger.info(args)

    emb_type = args.emb_type
    task_name = args.task_name

    assert emb_type in ['ica', 'pca']
    assert task_name in ['Google', 'MSR']

    # seed
    np.random.seed(0)

    analogy_tasks = {
        "Google": fetch_google_analogy(),
        "MSR": fetch_msr_analogy()
    }

    emb_path = "output/embeddings/glove_dic_and_emb.pkl"
    with open(emb_path, 'rb') as f:
        word2id, id2word, embed, pca_embed, ica_embed = pkl.load(f)

    ica_embed = pos_direct(ica_embed)
    skewness = np.mean(ica_embed**3, axis=0)
    skew_sort_idx = np.argsort(-skewness)
    ica_embed = ica_embed[:, skew_sort_idx]

    if emb_type == 'pca':
        embed = pca_embed
    elif emb_type == 'ica':
        embed = ica_embed

    data = []
    ps = [1, 2, 5, 10, 20, 50, 100, 200, 300]

    w = AnalogyEmbedding.from_words_and_vectors(
        id2word, embed, ps)

    # analogy tasks
    task = analogy_tasks[task_name]
    
    category_set = sorted(list(set(task.category)))
    for c in category_set:
        ids = np.where(task.category == c)[0]
        X, y = task.X[ids], task.y[ids]
        category = task.category[ids]
        p2res = evaluate_ps_analogy(w=w, X=X, y=y, category=category)
        for p in ps:
            res = p2res[p]
            acc = dict(res.loc[c])['accuracy']
            row = {
                'emb_type': emb_type,
                'p': p,
                'task_type': 'analogy',
                'task': c,
                'top1-acc': acc,
            }
            logger.info(row)
            data.append(row)

    # save
    df = pd.DataFrame(data)
    output_dir = Path("output/evaluation")
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / f'{emb_type}_{task_name}_analogy.csv'
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()