from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_data_file(input_file):
    """
    read the data file with a pickle format
    :param input_file: input path, string
    :return: the file's content
    """
    return pd.read_csv(input_file, header=0, encoding='utf-8')


def load_lm():
    """
    load bert's language model
    :return: the model and its corresponding tokenizer
    """
    model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, 'bert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights).to(DEVICE)
    return model, tokenizer


def tokenize(tokenizer, data, group):
    """
    Iterate over the data and tokenize it. Sequences longer than 512 tokens are trimmed.
    :param tokenizer: tokenizer to use for tokenization
    :param data: data to tokenize
    :return: a list of the entire tokenized data
    """
    tokenized_data = []
    for i, row in tqdm(data.iterrows(), desc="BERT Tokenization"):
        tokens = tokenizer.encode(str(row[f"Sentence_{group}"]), add_special_tokens=True)
        # keeping a maximum length of bert tokens: 512
        tokenized_data.append(tokens[:512])
    return tokenized_data


def encode_text(model, data):
    """
    encode the text
    :param model: encoding model
    :param data: data
    :return: two numpy matrices of the data:
                first: average of all tokens in each sentence
                second: cls token of each sentence
    """
    all_data_cls = []
    all_data_avg = []
    batch = []
    for row in tqdm(data, desc="BERT Encoding"):
        batch.append(row)
        input_ids = torch.tensor(batch).to(DEVICE)
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]
            all_data_avg.append(last_hidden_states.squeeze(0).mean(dim=0).cpu().numpy())
            all_data_cls.append(last_hidden_states.squeeze(0)[0].cpu().numpy())
        batch = []
    return np.array(all_data_avg), np.array(all_data_cls)


@timer
def main():
    parser = ArgumentParser()
    parser.add_argument("--treatment", type=str, default="gender", choices=("gender", "race"),
                        help="Specify treatment for experiments: gender, race")
    parser.add_argument("--corpus_type", type=str, default="enriched_noisy",
                        choices=("", "enriched", "enriched_noisy", "enriched_full"),
                        help="Corpus type can be: '', enriched, enriched_noisy, enriched_full")
    args = parser.parse_args()

    if args.corpus_type:
        treatment = f"{args.treatment}_{args.corpus_type}"
    else:
        treatment = args.treatment

    if args.treatment.startswith("gender"):
        data_path = Path(POMS_GENDER_DATASETS_DIR)
        output_path = Path(POMS_GENDER_DATA_DIR)
    else:
        data_path = Path(POMS_RACE_DATASETS_DIR)
        output_path = Path(POMS_RACE_DATA_DIR)

    output_path = output_path / "baselines" / "bert_encodings"
    output_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_lm()

    for exp in ("", "_bias_gentle_3", "_bias_aggressive_3"):
        for split in ("train", "dev", "test"):
            for group in ("F", "CF"):
                input_file = f"{treatment}{exp}_{split}.csv"
                print(f"Encoding data from: {input_file}")

                data = read_data_file(data_path / input_file)
                tokens = tokenize(tokenizer, data, group)

                avg_data, cls_data = encode_text(model, tokens)

                np.save(output_path / f"{treatment}{exp}_{split}_{group}_avg.npy", avg_data)
                np.save(output_path / f"{treatment}{exp}_{split}_{group}_cls.npy", cls_data)


if __name__ == '__main__':
    main()
