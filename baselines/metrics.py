from collections import defaultdict, Counter
from typing import Union, Tuple

import numpy as np


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
