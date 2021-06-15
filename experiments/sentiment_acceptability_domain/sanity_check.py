# Check that the model forgot the TC and remembered MLM and CC
import sys

import numpy as np
from datasets import load_dataset, load_metric

from transformers import TrainingArguments, BertTokenizer, Trainer, AutoModelForSequenceClassification, BertModel
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from experiments.sentiment_acceptability_domain.dataset import CaribbeanDataset
from utils import DATA_DIR, PROJECT_DIR


def training_pipeline(
        train_dataset,
        test_dataset,
        model_checkpoint='bert-base-uncased',
        num_labels=2,
        metric_name='accuracy',
        lr=2e-5,
        epochs=10,
        overfit=False,
        bert_intervention=None):
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint)

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    if bert_intervention:
        model.bert = bert_intervention

    metric = load_metric(metric_name)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labels)

    # noinspection PyTypeChecker
    args = TrainingArguments(
        'sanity_check',
        evaluation_strategy='epoch',
        learning_rate=lr,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        save_strategy='no'
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset if overfit else test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n============ before training ============")
    print(trainer.evaluate())
    trainer.train()
    print("\n============ after training ============")
    print(trainer.evaluate())


def main_cola():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # data
    cola = load_dataset('glue', 'cola')
    preprocess_function = lambda examples: tokenizer(examples['sentence'], truncation=True)
    cola = cola.map(preprocess_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labels)

    # noinspection PyTypeChecker
    args = TrainingArguments(
        'sanity_check',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=cola['train'],
        eval_dataset=cola['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n============before:")
    print(trainer.evaluate())
    print("\n============training:")
    trainer.train()
    print("\n============after:")
    print(trainer.evaluate())


if __name__ == '__main__':
    training_pipeline(
        train_dataset=CaribbeanDataset(data_path=f'{DATA_DIR}/acceptability_sample.csv', fold='train'),
        test_dataset=CaribbeanDataset(data_path=f'{DATA_DIR}/acceptability_sample.csv', fold='test'),
        overfit=False,
        bert_intervention=BertModel.from_pretrained(f'{PROJECT_DIR}/saved_models/SAD__2021_06_15__09_31_17')
    )
