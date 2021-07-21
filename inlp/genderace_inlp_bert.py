import random
import warnings
from argparse import ArgumentParser
from collections import defaultdict, Counter
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd
from POMS_GendeRace.datasets.create_poms_datasets import LABELS as POMS_LABELS
from Timer import timer
from baselines.inlp.nullspace_projection.src.debias import get_debiasing_projection
from constants import POMS_GENDER_DATASETS_DIR, POMS_GENDER_DATA_DIR, POMS_RACE_DATASETS_DIR, POMS_RACE_DATA_DIR, POMS_EXPERIMENTS_DIR
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC

from utils import init_logger, GoogleDriveHandler

warnings.filterwarnings("ignore")


# matplotlib.rcParams['agg.path.chunksize'] = 10000

# STOPWORDS = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])


### Load original data and pre-calculated BERT CLS states
def load_data(data_path: Path, encodings_path: Path, label_column: str, treatment: str, control: str, group: str, corpus_type: str,
              experiment: str = ""):
    data = dict()
    for split in ("train", "dev", "test"):
        if split == "test":
            input_file = f"{treatment}_{corpus_type}"
        else:
            input_file = f"{treatment}_{corpus_type}{experiment}"
        split_data = pd.read_csv(data_path / f"{input_file}_{split}.csv", header=0, encoding='utf-8')
        x_split = np.load(encodings_path / f"{input_file}_{split}_{group}_cls.npy")
        y_split = split_data[label_column].astype(int).to_numpy()
        z_treatment_split = split_data[f"{treatment.capitalize()}_{group}_label"].astype(int).to_numpy()
        z_control_split = split_data[f"{control.capitalize()}_label"].astype(int).to_numpy()
        assert len(split_data) == len(x_split) == len(y_split) == len(z_treatment_split) == len(z_control_split)
        data[split] = {"x": x_split, "y": y_split, f"z_{treatment}": z_treatment_split, f"z_{control}": z_control_split, "df": split_data}
    return data


### Train a POMS classifier
@timer
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


@timer
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


def get_TPR(y_pred, y_true, z_treatment):
    scores = defaultdict(Counter)
    poms_count_total = defaultdict(Counter)

    for y_hat, y, g in zip(y_pred, y_true, z_treatment):

        if y == y_hat:
            scores[y][g] += 1

        poms_count_total[y][g] += 1

    tprs = defaultdict(dict)
    tprs_change = dict()
    tprs_ratio = []

    for pom, scores_dict in scores.items():
        good_m, good_f = scores_dict[0], scores_dict[1]
        pom_total_f = poms_count_total[pom][1]
        pom_total_m = poms_count_total[pom][0]
        tpr_m = (good_m) / pom_total_m
        tpr_f = (good_f) / pom_total_f

        tprs[pom][0] = tpr_m
        tprs[pom][1] = tpr_f
        tprs_ratio.append(0)
        tprs_change[pom] = tpr_f - tpr_m

    return tprs, tprs_change, np.mean(np.abs(tprs_ratio))


def get_FPR2(y_pred, y_true, p2i, i2p, y_gender):
    fp = defaultdict(Counter)
    neg_count_total = defaultdict(Counter)
    pos_count_total = defaultdict(Counter)

    label_set = set(y_true)
    # count false positive per gender & class

    for y_hat, y, g in zip(y_pred, y_true, y_gender):

        if y != y_hat:
            fp[y_hat][g] += 1  # count false positives for y_hat

    # count total falses per gender (conditioned on class)

    total_prof_g = defaultdict(Counter)

    # collect POSITIVES for each profession and gender

    for y, g in zip(y_true, y_gender):
        total_prof_g[y][g] += 1

    total_m = sum([total_prof_g[y]["m"] for y in label_set])
    total_f = sum([total_prof_g[y]["f"] for y in label_set])

    # calculate NEGATIVES for each profession and gender

    total_false_prof_g = defaultdict(Counter)
    for y in label_set:
        total_false_prof_g[y]["m"] = total_m - total_prof_g[y]["m"]
        total_false_prof_g[y]["f"] = total_f - total_prof_g[y]["f"]

    fprs = defaultdict(dict)
    fprs_diff = dict()

    for profession, false_pred_dict in fp.items():
        false_male, false_female = false_pred_dict["m"], false_pred_dict["f"]
        prof_total_false_for_male = total_false_prof_g[profession]["m"]
        prof_total_false_for_female = total_false_prof_g[profession]["f"]

        ftr_m = false_male / prof_total_false_for_male
        ftr_f = false_female / prof_total_false_for_female
        fprs[i2p[profession]]["m"] = ftr_m
        fprs[i2p[profession]]["f"] = ftr_f
        fprs_diff[i2p[profession]] = ftr_m - ftr_f

    return fprs, fprs_diff


@timer
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


# def get_ATE(original_clf_f: LogisticRegression, original_f_x: np.ndarray, original_cf_x: np.ndarray) -> float:
#     """
#         Calculates ATE, given a classifier, factual and counterfactual test examples.
#
#         Args:
#             original_clf_f: A LogisticRegression classifier trained on factual training examples.
#             original_f_x: Vector representations for all factual test examples.
#                           Array shape (n_examples, vec_dim)
#             original_cf_x: Vector representations for all counterfactual test examples.
#                            Array shape (n_examples, vec_dim)
#
#         Returns:
#             ATE score (float).
#     """
#     original_clf_f_probs = original_clf_f.predict_proba(original_f_x)
#     original_clf_cf_probs = original_clf_f.predict_proba(original_cf_x)
#     return np.absolute(original_clf_f_probs - original_clf_cf_probs).sum(axis=1).mean()


def get_ATE(original_clf_f_probs: np.ndarray, original_clf_cf_probs: np.ndarray, confidence_intervals: bool = False) -> Union[
    float, Tuple[float, Tuple[float, float]]]:
    """
        Calculates ATE, given class probabilities predicted by the original classifier for all factual and counterfactual test examples.
        Args:
            original_clf_f_probs: Class probabilities for all factual test examples,
                                  predicted by classifier utilizing original vector representations.
                                  Array shape (n_examples, n_classes)
            original_clf_cf_probs: Class probabilities for all counterfactual test examples,
                                   predicted by classifier utilizing original vector representations.
                                   Array shape (n_examples, n_classes)
        Returns:
            ATE score (float).
    """
    ITE_array = np.absolute(original_clf_f_probs - original_clf_cf_probs).sum(axis=1)
    ATE = ITE_array.mean()
    if confidence_intervals:
        return ATE, get_confidence_intervals(ITE_array, ATE)
    else:
        return ATE


def get_TReATE(original_clf_probs: np.ndarray, treated_clf_probs: np.ndarray, confidence_intervals: bool = False) -> Union[
    float, Tuple[float, Tuple[float, float]]]:
    """
        Calculates TReATE, given class probabilities predicted by classifiers
        utilizing original and treated vector representations for all test examples.
        Args:
            original_clf_probs: Class probabilities for all test examples,
                                predicted by classifier utilizing original vector representations.
                                Array shape (n_examples, n_classes)
            treated_clf_probs: Class probabilities for all test examples,
                               predicted by classifier utilizing treated vector representations.
                               Array shape (n_examples, n_classes)
        Returns:
            TReATE score (float).
    """
    TRITE_array = np.absolute(original_clf_probs - treated_clf_probs).sum(axis=1)
    TReATE = TRITE_array.mean()
    if confidence_intervals:
        return TReATE, get_confidence_intervals(TRITE_array, TReATE)
    else:
        return TReATE


def get_CONEXP(original_clf_y_probs: np.ndarray, z_treatment_labels: np.ndarray, confidence_intervals: bool = False) -> Union[
    float, Tuple[float, Tuple[float, float]]]:
    """
        Calculates CONEXP, given y class probabilities predicted by original classifier and z_treatment labels for all test examples.
        Args:
            original_clf_y_probs: Class probabilities for all test examples,
                                  predicted by classifier utilizing original vector representations.
                                  Array shape (n_examples, n_classes)
            z_treatment_labels: Class probabilities for all test examples,
                               predicted by classifier utilizing treated vector representations.
                               Array shape (n_examples,)
        Returns:
            CONEXP score (float).
    """
    CONEXP_array = np.absolute(
        original_clf_y_probs[z_treatment_labels == 1, :].mean(axis=0) - original_clf_y_probs[z_treatment_labels == 0, :].mean(axis=0))
    CONEXP = CONEXP_array.sum()
    if confidence_intervals:
        return CONEXP, get_confidence_intervals(CONEXP_array, CONEXP)
    else:
        return CONEXP


def get_TPR_GAP(y_pred: np.ndarray, y_true: np.ndarray, z_treatment: np.ndarray) -> float:
    scores = defaultdict(Counter)
    poms_count_total = defaultdict(Counter)

    for y_hat, y, g in zip(y_pred, y_true, z_treatment):

        if y == y_hat:
            scores[y][g] += 1

        poms_count_total[y][g] += 1

    tprs = defaultdict(dict)
    tprs_change = dict()

    for pom, scores_dict in scores.items():
        good_m, good_f = scores_dict[0], scores_dict[1]
        pom_total_f = poms_count_total[pom][1]
        pom_total_m = poms_count_total[pom][0]
        tpr_m = (good_m) / pom_total_m
        tpr_f = (good_f) / pom_total_f

        tprs[pom][0] = tpr_m
        tprs[pom][1] = tpr_f
        tprs_change[pom] = tpr_f - tpr_m
    return np.absolute(list(tprs_change.values())).sum()


def get_confidence_intervals(results_array: np.ndarray, final_result: float) -> Tuple[float, float]:
    results_interval = 1.96 * results_array.std() / np.sqrt(len(results_array))
    return final_result - results_interval, final_result + results_interval


@timer
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
        encodings_path = Path(encodings_path) / "inlp" / "bert_encodings"
        treatment_output_path = Path(f"{POMS_EXPERIMENTS_DIR}/inlp/{args.version}") / treatment
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

        np.save(treatment_output_path / f"P_matrix.npy", P)
        np.save(treatment_output_path / f"rowspace_projections.npy", rowspace_projections)
        np.save(treatment_output_path / f"Ws_matrix.npy", Ws)

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
                np.save(exp_output_path / f"original_{task_type}_{task_variable}_{group}_test_predictions.npy", y_test_predictions)
                np.save(exp_output_path / f"original_{task_type}_{task_variable}_{group}_test_prediction_probs.npy", y_test_prediction_probs)

                clf_P, P_y_test_predictions, P_y_test_prediction_probs = finetune_downstream_classifier(P, train_xy, test_xy, logger)
                np.save(exp_output_path / f"inlp_projected_{task_type}_{task_variable}_{group}_test_predictions.npy", P_y_test_predictions)
                np.save(exp_output_path / f"inlp_projected_{task_type}_{task_variable}_{group}_test_prediction_probs.npy", P_y_test_prediction_probs)

                logger.info(
                    f"- Predicting Before and After Classifiers for {task_type.capitalize()} Task {task_variable} on CF data -")
                cf_clf_pred, cf_clf_pred_probs = predict_trained_classifier(clf,
                                                                            cf_data["test"]["x"],
                                                                            cf_data["test"][task_variable],
                                                                            logger)
                np.save(exp_output_path / f"original_{task_type}_{task_variable}_CF_test_predictions.npy", cf_clf_pred)
                np.save(exp_output_path / f"original_{task_type}_{task_variable}_CF_test_prediction_probs.npy", cf_clf_pred_probs)
                cf_clf_P_pred, cf_clf_P_pred_probs = predict_trained_classifier(clf_P,
                                                                                cf_data["test"]["x"],
                                                                                cf_data["test"][task_variable],
                                                                                logger, P)
                np.save(exp_output_path / f"inlp_projected_{task_type}_{task_variable}_CF_test_predictions.npy", cf_clf_P_pred)
                np.save(exp_output_path / f"inlp_projected_{task_type}_{task_variable}_CF_test_prediction_probs.npy", cf_clf_P_pred_probs)

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
