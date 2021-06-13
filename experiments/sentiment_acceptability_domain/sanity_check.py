# Check that the model forgot the TC and remembered MLM and CC
import argparse
import sys

import numpy as np
from datasets import load_metric

sys.path.append('/home/eldar.a/CausalEvaluation')

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, BertConfig, TrainingArguments, BertTokenizer, Trainer
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertModel

from experiments.sentiment_acceptability_domain.dataset import CaribbeanDataset
from models.BERT.configuration_causalm import BertCausalmConfig
from utils import DATA_DIR, PROJECT_DIR
from utils.metrics import calc_accuracy_from_logits


def main():
    # data
    train_dataset = CaribbeanDataset(data_path=f'{DATA_DIR}/acceptability_sample.csv', fold='train')
    test_dataset = CaribbeanDataset(data_path=f'{DATA_DIR}/acceptability_sample.csv', fold='test')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # config
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 2
    config.id2label = {0: 'unacceptable', 1: 'acceptable'}
    config.label2id = {'unacceptable': 0, 'acceptable': 1}
    config.problem_type = 'single_label_classification'

    def model_init():
        return BertForSequenceClassification(config)

    # noinspection PyTypeChecker
    args = TrainingArguments(
        'sanity_check',
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labels)

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n============before:")
    print(np.argmax(trainer.predict(test_dataset=test_dataset).predictions, axis=1))

    trainer.train()

    print("\n============after:")
    print(np.argmax(trainer.predict(test_dataset=test_dataset).predictions, axis=1))


if __name__ == '__main__':
    main()
