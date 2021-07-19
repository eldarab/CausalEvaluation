import warnings

import numpy as np
from sklearn.metrics import accuracy_score
from torch import no_grad
from torch import softmax, mean
from torch import abs as pt_abs
from torch import sum as pt_sum
from transformers import BertTokenizerFast

from utils import BERT_MODEL_CHECKPOINT, DEVICE


def calc_accuracy_from_logits(outputs, true_labels, model):
    logits = outputs.cpu().numpy()
    scores = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    predictions = [{"label": item.argmax(), "score": item.max().item()} for item in scores]
    accuracy = accuracy_score(y_true=true_labels.cpu(), y_pred=[item['label'] for item in predictions])
    return accuracy, predictions


class CausalmMetrics:
    def __init__(self):
        # TODO dataloader instead of dataset iterator
        pass

    @staticmethod
    def __compute_class_expectation(model, dataset, cls):
        model.eval()
        with no_grad():
            expectation = 0.
            for example in dataset:
                outputs = model(input_ids=example['input_ids'], attention_mask=example['attention_mask'], token_type_ids=example['token_type_ids'])
                cls_probabilities = softmax(outputs.logits, dim=-1)
                expectation += mean(cls_probabilities, dim=0)[cls].item()  # / len(dataloader)
            return expectation / len(dataset)

    @staticmethod
    def conexp(model, dataset, tc_indicator_name, cls=0) -> float:
        # TODO generalize to non-binary concept indicator
        dataset_0 = dataset.filter(lambda example: example[tc_indicator_name] == 0)
        dataset_1 = dataset.filter(lambda example: example[tc_indicator_name] == 1)

        e_0 = CausalmMetrics.__compute_class_expectation(model, dataset_0, cls)
        e_1 = CausalmMetrics.__compute_class_expectation(model, dataset_1, cls)
        conexp = abs(e_1 - e_0)

        return conexp

    @staticmethod
    def treate(model_o, model_cf, dataset, cls=0) -> float:
        model_o.eval()
        model_cf.eval()

        with no_grad():
            treate = 0.
            for example in dataset:
                outputs_o = model_o(input_ids=example['input_ids'], attention_mask=example['attention_mask'],
                                    token_type_ids=example['token_type_ids'])
                outputs_cf = model_cf(input_ids=example['input_ids'], attention_mask=example['attention_mask'],
                                      token_type_ids=example['token_type_ids'])

                cls_probabilities_o = softmax(outputs_o.logits, dim=-1)
                cls_probabilities_cf = softmax(outputs_cf.logits, dim=-1)

                treate += pt_abs(cls_probabilities_o - cls_probabilities_cf)[:, cls].sum().item()

            treate /= len(dataset)

        return treate

    @staticmethod
    def ate(model, dataset_f, dataset_cf, cls=0) -> float:
        assert len(dataset_f) == len(dataset_cf)
        model.eval()

        with no_grad():
            ate = 0.
            for example_f, example_cf in zip(dataset_f, dataset_cf):
                outputs_f = model(input_ids=example_f['input_ids'], attention_mask=example_f['attention_mask'],
                                  token_type_ids=example_f['token_type_ids'])
                outputs_cf = model(input_ids=example_cf['input_ids'], attention_mask=example_cf['attention_mask'],
                                   token_type_ids=example_cf['token_type_ids'])

                cls_probabilities_f = softmax(outputs_f.logits, dim=-1)
                cls_probabilities_cf = softmax(outputs_cf.logits, dim=-1)

                ate += pt_abs(cls_probabilities_f - cls_probabilities_cf)[:, cls].sum().item()

            ate /= len(dataset_f)

        return ate
