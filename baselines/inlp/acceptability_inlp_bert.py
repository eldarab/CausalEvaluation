from argparse import Namespace
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from baselines.inlp.src.debias import get_debiasing_projection
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC

from utils import DATA_DIR, PROJECT_DIR

ENCODED_DATA_DIR = PROJECT_DIR / 'baselines' / 'inlp' / 'bert_encodings'


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


def load_data(raw_data_path, encoded_data_path, task_label_col, treated_label_col, test_size=0.25):
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


def main():
    # get arguments
    args = Namespace(
        raw_data_filename='SAD_train.csv',
        encoded_data_filename='SAD_train__cls.npy',
    )

    # get paths
    raw_data_path = str(DATA_DIR / args.raw_data_filename)
    encoded_data_path = str(ENCODED_DATA_DIR / args.encoded_data_filename)

    # load data
    data = load_data(raw_data_path, encoded_data_path, task_label_col='sentiment', treated_label_col='acceptability_amitavasil')

    # run INLP
    P, _, _ = get_projection_matrix(**data)

    # compute TReATE
    control_classifier = train_task_classifier(data['X_train'], data['Y_train'])
    treated_classifier = train_task_classifier(data['X_train'], data['Y_train'], P)
    treate_inlp = np.abs(treated_classifier.predict_proba(data['X_test']) - control_classifier.predict_proba(data['X_test'])).mean()
    print(f'TReATE INLP: {treate_inlp:.3f}')


if __name__ == '__main__':
    main()
