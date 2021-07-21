from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from utils import DEVICE, BERT_MODEL_CHECKPOINT, PROJECT_DIR, DATA_DIR


def load_lm():
    """
    load bert's language model
    :return: the model and its corresponding tokenizer
    """
    model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, BERT_MODEL_CHECKPOINT)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights).to(DEVICE)
    return model, tokenizer


def load_data(data_path):
    return pd.read_csv(data_path, index_col=0)


def tokenize(tokenizer: BertTokenizer, data: pd.DataFrame, text_key='text'):
    """
    Iterate over the data and tokenize it. Sequences longer than 512 tokens are trimmed.
    :param tokenizer: tokenizer to use for tokenization
    :param data: data to tokenize
    :param text_key:
    :return: a list of the entire tokenized data
    """
    tokenized_data = []
    for i, row in tqdm(data.iterrows(), desc="BERT Tokenization"):
        tokens = tokenizer.encode(str(row[text_key]), add_special_tokens=True)
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


def main():
    # get arguments
    args = Namespace(
        data_filename='SAD_train.csv',
        text_key='text',
    )

    # get paths
    data_path = str(DATA_DIR / args.data_filename)
    output_path = PROJECT_DIR / 'baselines' / 'inlp' / 'bert_encodings'
    output_path.mkdir(parents=True, exist_ok=True)
    output_filename = args.data_filename.split('.')[0]

    # encode
    model, tokenizer = load_lm()
    data = load_data(data_path)
    tokens = tokenize(tokenizer, data, args.text_key)
    avg_data, cls_data = encode_text(model, tokens)

    # save encodings
    avg_data_filename = f"{output_filename}__avg.npy"
    cls_data_filename = f"{output_filename}__cls.npy"
    np.save(str(output_path / avg_data_filename), avg_data)
    np.save(str(output_path / cls_data_filename), cls_data)
    print(f'Saved "{avg_data_filename}" and "{cls_data_filename}" under {str(output_path)}.')


if __name__ == '__main__':
    main()
