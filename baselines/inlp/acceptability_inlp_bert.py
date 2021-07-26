from argparse import Namespace
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from baselines.inlp.src.debias import get_debiasing_projection
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC

from utils import DATA_DIR, PROJECT_DIR, BERT_MODEL_CHECKPOINT, DEVICE

ENCODED_DATA_DIR = PROJECT_DIR / 'baselines' / 'inlp' / 'bert_encodings'


def load_lm():
    """
    load bert's language model
    :return: the model and its corresponding tokenizer
    """
    model_class, tokenizer_class, pretrained_weights = (BertModel, BertTokenizer, BERT_MODEL_CHECKPOINT)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights).to(DEVICE)
    return model, tokenizer


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


def encode_bert_states():
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
    data = pd.read_csv(data_path, index_col=0)
    tokens = tokenize(tokenizer, data, args.text_key)
    avg_data, cls_data = encode_text(model, tokens)

    # save encodings
    avg_data_filename = f"{output_filename}__avg.npy"
    cls_data_filename = f"{output_filename}__cls.npy"
    np.save(str(output_path / avg_data_filename), avg_data)
    np.save(str(output_path / cls_data_filename), cls_data)
    print(f'Saved "{avg_data_filename}" and "{cls_data_filename}" under {str(output_path)}.')


def get_projection_matrix(X_train, Z_train, Y_train, X_test, Z_test, Y_test):
    P, rowspace_projections, Ws = get_debiasing_projection(
        classifier_class=SGDClassifier,
        classifier_params={'loss': 'hinge', 'penalty': 'l2', 'fit_intercept': False, 'class_weight': None, 'n_jobs': -1},
        num_classifiers=3,
        input_dim=768,
        is_autoregressive=True,
        min_accuracy=0.,
        X_train=X_train,
        Y_train=Z_train,
        X_dev=X_test,
        Y_dev=Z_test,
        Y_train_main=Y_train,
        Y_dev_main=Y_test,
        by_class=True
    )
    return P, rowspace_projections, Ws


def load_encoded_data(raw_data_path, encoded_data_path, task_label_col, treated_label_col, test_size=0.25):
    raw_df = pd.read_csv(raw_data_path, index_col=0)[[task_label_col, treated_label_col]]
    raw_df = raw_df.rename(columns={task_label_col: 'Y', treated_label_col: 'Z'})
    X = np.load(encoded_data_path)
    Y = raw_df['Y']
    Z = raw_df['Z']

    data = dict()
    data['X_train'], data['X_test'], data['Y_train'], data['Y_test'], data['Z_train'], data['Z_test'] = train_test_split(X, Y, Z, test_size=test_size)

    return data


def train_task_classifier(X, y, P=None):
    clf = LogisticRegression(warm_start=True, penalty='l2', solver="sag", multi_class='multinomial', fit_intercept=True, verbose=10, max_iter=7,
                             n_jobs=-1, random_state=1)
    if P is not None:
        clf.fit(X @ P, y)
    else:
        clf.fit(X, y)

    return clf


def acceptability_inlp_bert():
    # get arguments
    args = Namespace(
        raw_data_filename='SAD_train.csv',
        encoded_data_filename='SAD_train__cls.npy',
    )

    # get paths
    raw_data_path = str(DATA_DIR / args.raw_data_filename)
    encoded_data_path = str(ENCODED_DATA_DIR / args.encoded_data_filename)

    # load data
    data = load_encoded_data(raw_data_path, encoded_data_path, task_label_col='sentiment', treated_label_col='acceptability_amitavasil')

    # run INLP
    P, _, _ = get_projection_matrix(**data)

    # compute TReATE
    control_classifier = train_task_classifier(data['X_train'], data['Y_train'])
    treated_classifier = train_task_classifier(data['X_train'], data['Y_train'], P)
    treate_inlp = np.abs(treated_classifier.predict_proba(data['X_test']) - control_classifier.predict_proba(data['X_test'])).mean()
    print(f'TReATE INLP: {treate_inlp:.3f}')


if __name__ == '__main__':
    encode_bert_states()
    acceptability_inlp_bert()
