import argparse
import logging
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from utils import (get_batch_results, get_batch_size, get_model_names,
                   get_sentences)

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Save contextualized embeddings.")

    parser.add_argument("--max_same_token", type=int, default=10)
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(args)
    max_same_token = args.max_same_token
    vocab_size = args.vocab_size
    data_path = args.data_path

    output_dir = Path("output/embeddings")
    output_dir.mkdir(exist_ok=True, parents=True)

    sentences = get_sentences(data_path)
    logger.info(f"number of sentences: {len(sentences)}")

    encoder_model_names, decoder_model_names = get_model_names()
    model_names = encoder_model_names + decoder_model_names

    model2token2sentence = dict()
    for model_name in model_names:
        logger.info(f"model_name: {model_name}")

        if model_name == "bert-base-uncased":
            prefix = ""
        else:
            prefix = "Ġ"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name in encoder_model_names:
            model = AutoModel.from_pretrained(model_name).to("cuda")
        elif model_name in decoder_model_names:
            model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        model.eval()

        # check sentences contain cased words
        batch_size = get_batch_size(model_name)
        logger.info(f"batch_size: {batch_size}")

        token2count = defaultdict(int)
        token2sentence = dict()

        token_list = []
        embedding_list = []
        total_count = 0
        end_flag = False
        for batch_index in tqdm(
            list(range((len(sentences) + batch_size - 1) // batch_size))
        ):
            batch_sentences = sentences[
                batch_index * batch_size : (batch_index + 1) * batch_size
            ]

            batch_tokens, batch_indexs, batch_segments_ids = get_batch_results(
                batch_sentences, tokenizer
            )

            indexs_tensor = torch.tensor(batch_indexs).to("cuda")
            segments_tensors = torch.tensor(batch_segments_ids).to("cuda")

            with torch.no_grad():
                try:
                    outputs = model(
                        input_ids=indexs_tensor,
                        attention_mask=segments_tensors,
                        output_hidden_states=True,
                    )
                except Exception as e:
                    logger.info(e)
                    continue
                batch_embeddings = outputs.hidden_states[-1]

            del outputs

            batch_embeddings = batch_embeddings.to("cpu").numpy()

            for indexs, tokens, embeddings, sentence in zip(
                batch_indexs, batch_tokens, batch_embeddings, batch_sentences
            ):
                assert len(embeddings) >= len(tokens)
                indexs = indexs[: len(tokens)]
                embeddings = embeddings[: len(indexs)]

                for token, embedding in zip(tokens, embeddings):
                    if token == tokenizer.bos_token or token == tokenizer.cls_token:
                        continue
                    elif token == tokenizer.eos_token or token == tokenizer.sep_token:
                        continue

                    if token2count[token] >= max_same_token:
                        continue

                    token2count[token] += 1
                    token = f"{token}_{token2count[token] - 1}"

                    # NOTE: Strictly speaking, we should have considered the tokens
                    # with prefix (e.g., ##light_1) for BERT or without prefix
                    #  (e.g., "ultraviolet_1") for RoBERTa, GPT-2, Pythia-160m,
                    # but we forget to do so in the paper. :(
                    if token.startswith(prefix + "ultraviolet_") or token.startswith(
                        prefix + "light_"
                    ):
                        token2sentence[token] = sentence

                    token_list.append(token)
                    embedding_list.append(embedding)
                    total_count += 1
                    if total_count >= vocab_size:
                        end_flag = True
                        break
                if end_flag:
                    break
            if end_flag:
                break
            logger.info(f"total_count: {total_count}")

        model2token2sentence[model_name.replace("/", "-")] = token2sentence

        # check
        token_and_count = [(token, count) for token, count in token2count.items()]
        token_and_count = sorted(token_and_count, key=lambda x: -x[1])
        logger.info(f"top 10 tokens: {token_and_count[:10]}")
        logger.info(f"bottom 10 tokens: {token_and_count[-10:]}")

        assert len(token_list) == len(embedding_list) == total_count
        token_list = np.array(token_list)
        embedding_list = np.array(embedding_list)
        logger.info(f"embs_shape: {embedding_list.shape}")
        save_path = output_dir / f"{model_name.replace('/', '-')}.pkl"
        logger.info(f"save_path: {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump((token_list, embedding_list), f)

        output_dir2 = Path("output/data_for_ultraviolet_and_light_bargraphs")
        output_dir2.mkdir(exist_ok=True, parents=True)
        output_path2 = output_dir2 / "model2token2sentence.pkl"
        logger.info(f"data for ultraviolet and light bargraphs: {output_path2}")
        with open(output_path2, "wb") as f:
            pickle.dump(model2token2sentence, f)


if __name__ == "__main__":
    main()
