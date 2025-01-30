import logging
import pickle as pkl
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main(word):

    dynamic_models = [
        "bert-base-uncased",
        "roberta-base",
        "gpt2",
        "EleutherAI-pythia-160m",
    ]

    # Load embeddings
    output_dir = Path("output/data_for_ultraviolet_and_light_bargraphs/")
    input_path = output_dir / "model2token2sentence.pkl"
    with open(input_path, "rb") as f:
        model2token2sentence = pkl.load(f)

    # token2sentence check
    model2token_info = {}
    for model_name in dynamic_models:
        token2sentence = model2token2sentence[model_name]
        sent_count = defaultdict(int)
        token_info = []
        for token in sorted(token2sentence.keys()):
            sentence = token2sentence[token]
            if word in token:
                sent_count[sentence] += 1
                sent_c = sent_count[sentence]
                sentence = sentence + f"_{sent_c}"
                token_info.append((token, model_name, sentence))
        model2token_info[model_name] = token_info

    sentence2model_info = defaultdict(list)
    for model_name, token_info in model2token_info.items():
        for token, model_name, sentence in token_info:
            sentence2model_info[sentence].append((token, model_name))

    valid_sentences = []
    invalid_sentences = []
    for sentence, model_info in sentence2model_info.items():
        if len(model_info) != len(dynamic_models):
            invalid_sentences.append(sentence)
            logger.info(f"invalid sentence: {sentence}")
        else:
            valid_sentences.append(sentence)

    token_and_model2new_idx_and_sentence = {}
    for new_idx, sentence in enumerate(valid_sentences):
        model_info = sentence2model_info[sentence]
        for token, model_name in model_info:
            token_and_model2new_idx_and_sentence[(token, model_name)] = (
                new_idx,
                sentence,
            )

    # save valid token2sentence
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"token_and_model2new_idx_and_sentence_{word}.pkl"
    logger.info(f"output_path: {output_path}")
    with open(output_path, "wb") as f:
        pkl.dump(token_and_model2new_idx_and_sentence, f)


if __name__ == "__main__":
    main(word="light")
    main(word="ultraviolet")
