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
    def __init__(self, tokenizer_checkpoint, device=DEVICE):
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_checkpoint)
        self.device = device
        # todo dataloader instead of dataset iterator

    def __compute_class_expectation(self, model, dataset, cls=0):
        model.eval()
        with no_grad():
            expectation = 0.
            for example in dataset:
                inputs = self.tokenizer(example['text'], padding=True, truncation=True, return_tensors='pt')
                inputs.to(self.device)

                outputs = model(**inputs)

                probas = softmax(outputs.logits, dim=-1)

                expectation += mean(probas, dim=0)[cls].item()  # / len(dataloader)

            return expectation / len(dataset)

    def conexp(self, model, dataset) -> float:
        dataset_0 = dataset.filter(lambda example: example['tc_labels'] == 0)
        dataset_1 = dataset.filter(lambda example: example['tc_labels'] == 1)

        e_0 = self.__compute_class_expectation(model, dataset_0, cls=0)
        e_1 = self.__compute_class_expectation(model, dataset_1, cls=0)

        conexp = abs(e_1 - e_0)

        return conexp

    def treate(self, model_o, model_cf, dataset, cls=0) -> float:
        model_o.eval()
        model_cf.eval()

        with no_grad():
            treate = 0.
            for example in dataset:
                inputs = self.tokenizer(example['text'], padding=True, truncation=True, return_tensors='pt')
                inputs.to(self.device)

                outputs_o = model_o(**inputs)
                outputs_cf = model_cf(**inputs)

                probas_o = softmax(outputs_o.logits, dim=-1)
                probas_cf = softmax(outputs_cf.logits, dim=-1)

                treate += pt_abs(probas_o - probas_cf)[:, cls].sum().item()

            treate /= len(dataset)

        return treate

    def ate(self, model, dataset_f, dataset_cf, cls=0) -> float:
        assert len(dataset_f) == len(dataset_cf)
        model.eval()

        with no_grad():
            ate = 0.
            for example_f, example_cf in zip(dataset_f, dataset_cf):
                inputs_f = self.tokenizer(example_f['text'], padding=True, truncation=True, return_tensors='pt')
                inputs_cf = self.tokenizer(example_cf['text'], padding=True, truncation=True, return_tensors='pt')

                inputs_f.to(self.device)
                inputs_cf.to(self.device)

                outputs_f = model(**inputs_f)
                outputs_cf = model(**inputs_cf)

                probas_f = softmax(outputs_f.logits, dim=-1)
                probas_cf = softmax(outputs_cf.logits, dim=-1)

                ate += pt_abs(probas_f - probas_cf)[:, cls].sum().item()

            ate /= len(dataset_f)

        return ate
