import logging

import numpy as np
import scipy.stats

logger = logging.getLogger(__name__)


def pos_direct(vecs):
    vecs = vecs * np.sign(scipy.stats.skew(vecs, axis=0))
    return vecs


def get_model_names():
    encoder_model_names = [
        "bert-base-uncased",
        "roberta-base",
    ]
    decoder_model_names = [
        "gpt2",
        "EleutherAI/pythia-160m",
    ]

    return encoder_model_names, decoder_model_names


def get_batch_size(model_name):
    model_name2batch_size = {
        "bert-base-uncased": 512,
        "roberta-base": 512,
        "gpt2": 128,
        "EleutherAI/pythia-160m": 128,
    }

    return model_name2batch_size[model_name]


def get_batch_results(batch_sentences, tokenizer):
    # bos_token
    if tokenizer.bos_token is None:
        bos_token = tokenizer.cls_token
    else:
        bos_token = tokenizer.bos_token

    # eos_token
    if tokenizer.eos_token is None:
        eos_token = tokenizer.sep_token
    else:
        eos_token = tokenizer.eos_token

    batch_tokens = []
    batch_indexs = []
    batch_segments_ids = []

    for sentence in batch_sentences:
        # Model tokenization
        tokens = [bos_token] + tokenizer.tokenize(sentence) + [eos_token]
        batch_tokens.append(tokens)

        indexs = tokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(tokens)
        assert len(indexs) == len(segments_ids)

        batch_indexs.append(indexs)
        batch_segments_ids.append(segments_ids)

    batch_max_len = max([len(indexs) for indexs in batch_indexs])

    for indexs, segments_ids in zip(batch_indexs, batch_segments_ids):
        assert len(indexs) == len(segments_ids)
        indexs.extend([0] * (batch_max_len - len(indexs)))
        segments_ids.extend([0] * (batch_max_len - len(segments_ids)))

    return batch_tokens, batch_indexs, batch_segments_ids


def get_sentences(data_path):
    with open(data_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip().lower() for line in lines]

    # priortize ultraviolet sentences
    ultraviolet_sentences = []
    other_sentences = []
    for line in lines:
        if "ultraviolet" in line:
            ultraviolet_sentences.append(line)
        else:
            other_sentences.append(line)
    lines = ultraviolet_sentences + other_sentences
    return lines
