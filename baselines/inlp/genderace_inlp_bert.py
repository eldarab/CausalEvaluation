import random
import warnings
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC

from baselines.inlp.src.debias import get_debiasing_projection
from baselines.metrics import get_TPR, get_ATE, get_TReATE, get_CONEXP
from utils import init_logger, GoogleDriveHandler

warnings.filterwarnings("ignore")


# Load original data and pre-calculated BERT CLS states
def load_data(data_path: Path, encodings_path: Path, label_column: str, treatment: str, control: str, group: str, corpus_type: str,
              experiment: str = ""):
    data = dict()
    for split in ("train", "dev", "test"):
        if split == "test":
            input_file = f"{treatment}_{corpus_type}"
        else:
            input_file = f"{treatment}_{corpus_type}{experiment}"
        split_data = pd.read_csv(data_path / f"{input_file}_{split}.csv", header=0, encoding='utf-8')
        x_split = np.load(str(encodings_path / f"{input_file}_{split}_{group}_cls.npy"))
        y_split = split_data[label_column].astype(int).to_numpy()
        z_treatment_split = split_data[f"{treatment.capitalize()}_{group}_label"].astype(int).to_numpy()
        z_control_split = split_data[f"{control.capitalize()}_label"].astype(int).to_numpy()
        assert len(split_data) == len(x_split) == len(y_split) == len(z_treatment_split) == len(z_control_split)
        data[split] = {"x": x_split, "y": y_split, f"z_{treatment}": z_treatment_split, f"z_{control}": z_control_split, "df": split_data}
    return data


# Train a POMS classifier
def train_downstream_classifier(train_xy: Tuple[np.ndarray, np.ndarray],
                                test_xy: Tuple[np.ndarray, np.ndarray],
                                logger):
    x_train, y_train = train_xy
    x_test, y_test = test_xy
    random.seed(0)
    np.random.seed(0)

    clf = LogisticRegression(warm_start=True, penalty='l2',
                             solver="saga", multi_class='multinomial', fit_intercept=False,
                             verbose=5, n_jobs=-1, random_state=1, max_iter=7)

    # params = {'}
    # clf = SGDClassifier(loss= 'hinge', max_iter = 4000, fit_intercept= True, class_weight= None, n_jobs= 100)

    idx = np.random.rand(x_train.shape[0]) < 1.0
    clf.fit(x_train[idx], y_train[idx])
    logger.info(f"Classifier Train Accuracy: {clf.score(x_train, y_train)}")
    test_predictions, test_prediction_probs = predict_trained_classifier(clf, x_test, y_test, logger)
    return clf, test_predictions, test_prediction_probs


def predict_trained_classifier(clf, x_test, y_test, logger, P=None):
    if P is not None:
        projected_x_test = (P.dot(x_test.T)).T
        logger.info(f"Classifier Projected Test Accuracy: {clf.score(projected_x_test, y_test)}")
        test_predictions = clf.predict(projected_x_test)
        test_prediction_probs = clf.predict_proba(projected_x_test)
    else:
        logger.info(f"Classifier Test Accuracy: {clf.score(x_test, y_test)}")
        test_predictions = clf.predict(x_test)
        test_prediction_probs = clf.predict_proba(x_test)
    return test_predictions, test_prediction_probs


def get_projection_matrix(num_clfs, X_train, Z_train_treatment, X_dev, Z_dev_treatment, Y_train_task, Y_dev_task):
    is_autoregressive = True
    min_acc = 0.
    # noise = False
    dim = 768
    n = num_clfs
    # random_subset = 1.0
    TYPE = "svm"

    # if MLP:
    #     x_train_gender = np.matmul(x_train, clf.coefs_[0]) + clf.intercepts_[0]
    #     x_dev_gender = np.matmul(x_dev, clf.coefs_[0]) + clf.intercepts_[0]
    # else:
    #     x_train_gender = x_train.copy()
    #     x_dev_gender = x_dev.copy()

    if TYPE == "sgd":
        gender_clf = SGDClassifier
        params = {'loss': 'hinge', 'penalty': 'l2', 'fit_intercept': False, 'class_weight': None, 'n_jobs': -1}
    else:
        gender_clf = LinearSVC
        params = {'penalty': 'l2', 'C': 0.01, 'fit_intercept': True, 'class_weight': None, "dual": False}

    P, rowspace_projections, Ws = get_debiasing_projection(gender_clf, params, n, dim, is_autoregressive,
                                                           min_acc,
                                                           X_train, Z_train_treatment, X_dev, Z_dev_treatment,
                                                           Y_train_main=Y_train_task, Y_dev_main=Y_dev_task,
                                                           by_class=True)
    return P, rowspace_projections, Ws


def finetune_downstream_classifier(P, train_xy, test_xy, logger):
    x_train, y_train = train_xy
    x_test, y_test = test_xy

    clf = LogisticRegression(warm_start=True, penalty='l2',
                             solver="sag", multi_class='multinomial', fit_intercept=True,
                             verbose=10, max_iter=3, n_jobs=-1, random_state=1)
    # clf = SGDClassifier()
    P_rowspace = np.eye(768) - P
    mean_gender_vec = np.mean(P_rowspace.dot(x_train.T).T, axis=0)
    # 2
    projected_x_train = (P.dot(x_train.T)).T
    projected_x_test = (P.dot(x_test.T)).T

    clf.fit(projected_x_train, y_train)

    logger.info(f"Fine-tune Train Accuracy: {clf.score(projected_x_train, y_train)}")
    # print(clf.fit((x_train.T).T + mean_gender_vec, y_train))

    logger.info(f"Fine-tune Test Accuracy: {clf.score(projected_x_test, y_test)}")
    test_predictions = clf.predict(projected_x_test)
    test_prediction_probs = clf.predict_proba(projected_x_test)
    return clf, test_predictions, test_prediction_probs


def main():
    parser = ArgumentParser()
    parser.add_argument("--corpus_type", type=str, default="enriched_noisy",
                        choices=("", "enriched", "enriched_noisy", "enriched_full"),
                        help="Corpus type can be: '', enriched, enriched_noisy, enriched_full")
    parser.add_argument("--version", type=str, default="-1")
    args = parser.parse_args()

    poms_labels = {v: k for k, v in POMS_LABELS.items()}

    drive_handler = GoogleDriveHandler()

    for treatment, control, data_path, encodings_path, treatment_labels in zip(("gender", "race"), ("race", "gender"),
                                                                               (POMS_GENDER_DATASETS_DIR, POMS_RACE_DATASETS_DIR),
                                                                               (POMS_GENDER_DATA_DIR, POMS_RACE_DATA_DIR),
                                                                               (("Male", "Female"), ("European", "African-American"))):
        encodings_path = Path(encodings_path) / "baselines" / "bert_encodings"
        treatment_output_path = Path(f"{POMS_EXPERIMENTS_DIR}/baselines/{args.version}") / treatment
        logger = init_logger(f"{treatment}_experiments", str(treatment_output_path))

        # V1
        logger.info(f"- Calculating Projection Matrix for Treatment Task {treatment} -")
        f_data = load_data(Path(data_path), encodings_path, "POMS_label", treatment, control, "F", args.corpus_type)
        idx = np.random.rand(f_data["train"]["x"].shape[0]) < 1.
        P, rowspace_projections, Ws = get_projection_matrix(300,
                                                            f_data["train"]["x"][idx],
                                                            f_data["train"][f"z_{treatment}"][idx],
                                                            f_data["dev"]["x"],
                                                            f_data["dev"][f"z_{treatment}"],
                                                            f_data["train"]["y"],
                                                            f_data["dev"]["y"])

        np.save(str(treatment_output_path / f"P_matrix.npy"), P)
        np.save(str(treatment_output_path / f"rowspace_projections.npy"), rowspace_projections)
        np.save(str(treatment_output_path / f"Ws_matrix.npy"), Ws)

        push_message = drive_handler.push_files(str(treatment_output_path))
        logger.info(push_message)

        for experiment in ("", "_bias_gentle_3", "_bias_aggressive_3"):
            metrics_dict = dict()
            exp_output_path = treatment_output_path / f"{args.corpus_type}{experiment}"
            exp_output_path.mkdir(parents=True, exist_ok=True)

            group = "F"
            logger.info(f"\n---- Starting {treatment}_{args.corpus_type}{experiment} experiment for group {group} ----\n")
            f_data = load_data(Path(data_path), encodings_path, "POMS_label", treatment, control, "F", args.corpus_type, experiment)
            cf_data = load_data(Path(data_path), encodings_path, "POMS_label", treatment, control, "CF", args.corpus_type, experiment)

            # # V2
            # logger.info(f"- Calculating Projection Matrix for Treatment Task {treatment} on {'balanced' if not experiment else experiment} data -")
            # idx = np.random.rand(f_data["train"]["x"].shape[0]) < 1.
            # P, rowspace_projections, Ws = get_projection_matrix(300,
            #                                                     f_data["train"]["x"][idx],
            #                                                     f_data["train"][f"z_{treatment}"][idx],
            #                                                     f_data["dev"]["x"],
            #                                                     f_data["dev"][f"z_{treatment}"],
            #                                                     f_data["train"]["y"],
            #                                                     f_data["dev"]["y"])
            #
            # np.save(treatment_output_path / f"P_matrix.npy", P)
            # np.save(treatment_output_path / f"rowspace_projections.npy", rowspace_projections)
            # np.save(treatment_output_path / f"Ws_matrix.npy", Ws)

            for task_variable, task_type in zip((f"z_{treatment}", f"z_{control}", "y"), ("treatment", "control", "downstream")):
                logger.info(f"- Training Before and After Classifiers for {task_type.capitalize()} Task {task_variable} -")
                train_xy = (f_data["train"]["x"], f_data["train"][task_variable])
                test_xy = (f_data["test"]["x"], f_data["test"][task_variable])

                clf, y_test_predictions, y_test_prediction_probs = train_downstream_classifier(train_xy, test_xy, logger)
                np.save(str(exp_output_path / f"original_{task_type}_{task_variable}_{group}_test_predictions.npy"), y_test_predictions)
                np.save(str(exp_output_path / f"original_{task_type}_{task_variable}_{group}_test_prediction_probs.npy"), y_test_prediction_probs)

                clf_P, P_y_test_predictions, P_y_test_prediction_probs = finetune_downstream_classifier(P, train_xy, test_xy, logger)
                np.save(str(exp_output_path / f"inlp_projected_{task_type}_{task_variable}_{group}_test_predictions.npy"), P_y_test_predictions)
                np.save(str(exp_output_path / f"inlp_projected_{task_type}_{task_variable}_{group}_test_prediction_probs.npy"),
                        P_y_test_prediction_probs)

                logger.info(
                    f"- Predicting Before and After Classifiers for {task_type.capitalize()} Task {task_variable} on CF data -")
                cf_clf_pred, cf_clf_pred_probs = predict_trained_classifier(clf,
                                                                            cf_data["test"]["x"],
                                                                            cf_data["test"][task_variable],
                                                                            logger)
                np.save(str(exp_output_path / f"original_{task_type}_{task_variable}_CF_test_predictions.npy"), cf_clf_pred)
                np.save(str(exp_output_path / f"original_{task_type}_{task_variable}_CF_test_prediction_probs.npy"), cf_clf_pred_probs)
                cf_clf_P_pred, cf_clf_P_pred_probs = predict_trained_classifier(clf_P,
                                                                                cf_data["test"]["x"],
                                                                                cf_data["test"][task_variable],
                                                                                logger, P)
                np.save(str(exp_output_path / f"inlp_projected_{task_type}_{task_variable}_CF_test_predictions.npy"), cf_clf_P_pred)
                np.save(str(exp_output_path / f"inlp_projected_{task_type}_{task_variable}_CF_test_prediction_probs.npy"), cf_clf_P_pred_probs)

                if task_type == "downstream":
                    metrics_dict[f"original_F_clf"] = clf
                    metrics_dict[f"treated_F_clf"] = clf_P
                    metrics_dict[f"original_F_clf_pred"] = y_test_predictions
                    metrics_dict[f"treated_F_clf_pred"] = P_y_test_predictions
                    metrics_dict[f"original_F_clf_pred_probs"] = y_test_prediction_probs
                    metrics_dict[f"treated_F_clf_pred_probs"] = P_y_test_prediction_probs
                    metrics_dict[f"original_CF_clf_pred"] = cf_clf_pred
                    metrics_dict[f"original_CF_clf_pred_probs"] = cf_clf_pred_probs
                    metrics_dict[f"treated_CF_clf_pred"] = cf_clf_P_pred
                    metrics_dict[f"treated_CF_clf_pred_probs"] = cf_clf_P_pred_probs

            for group, group_data in zip(("F", "CF"), (f_data, cf_data)):
                logger.info(f"-- Results for {treatment}_{args.corpus_type}{experiment} experiment for group {group} --\n")
                tprs_before, tprs_change_before, mean_ratio_before = get_TPR(metrics_dict[f"original_{group}_clf_pred"],
                                                                             group_data["test"]["y"],
                                                                             group_data["test"][f"z_{treatment}"])
                tprs_after, tprs_change_after, mean_ratio_after = get_TPR(metrics_dict[f"treated_{group}_clf_pred"],
                                                                          group_data["test"]["y"],
                                                                          group_data["test"][f"z_{treatment}"])
                for label in tprs_before:
                    for treatment_label in tprs_before[label]:
                        logger.info(
                            f"TPR before - {poms_labels[label]} - {treatment_labels[treatment_label]}: {tprs_before[label][treatment_label]:.04f}")
                        logger.info(
                            f"TPR after - {poms_labels[label]} - {treatment_labels[treatment_label]}: {tprs_after[label][treatment_label]:.04f}")
                    logger.info(f"TPR Change before - {poms_labels[label]}: {tprs_change_before[label]:.04f}")
                    logger.info(f"TPR Change after - {poms_labels[label]}: {tprs_change_after[label]:.04f}")
                logger.info(f"Mean Ratio before: {mean_ratio_before:.04f}")
                logger.info(f"Mean Ratio after: {mean_ratio_after:.04f}\n")
                logger.info(f"TPR-GAP(O): {np.absolute(list(tprs_change_before.values())).sum():.04f}")
                logger.info(f"TPR-GAP(INLP): {np.absolute(list(tprs_change_after.values())).sum():.04f}\n")
                logger.info(
                    f"CONEXP(O): {get_CONEXP(metrics_dict[f'original_{group}_clf_pred_probs'], group_data['test'][f'z_{treatment}'], True)}\n")
                logger.info(
                    f"TReATE(O,INLP): {get_TReATE(metrics_dict[f'original_{group}_clf_pred_probs'], metrics_dict[f'treated_{group}_clf_pred_probs'], True)}\n")
            logger.info(f"ATE(O): {get_ATE(metrics_dict[f'original_F_clf_pred_probs'], metrics_dict[f'original_CF_clf_pred_probs'], True)}\n")

            push_message = drive_handler.push_files(str(exp_output_path))
            logger.info(push_message)
            # send_email(push_message, treatment)


if __name__ == "__main__":
    main()
