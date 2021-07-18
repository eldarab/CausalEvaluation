from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, TrainingArguments, Trainer
import numpy as np

from modeling.BERT.bert_causalm import BertForCausalmAdditionalPreTraining
from modeling.BERT.configuration_causalm import BertCausalmConfig, CausalmHeadConfig
from utils import TOKEN_CLASSIFICATION, DataCollatorForCausalmAdditionalPretraining, tokenize_and_align_labels


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[lbl] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def show_results():
    predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[pred] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[lbl] for (pred, lbl) in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return results


if __name__ == '__main__':
    model_checkpoint = "bert-base-uncased"
    batch_size = 16

    datasets = load_dataset("conll2003")
    datasets = datasets.rename_column('ner_tags', 'cc_labels')
    label_list = datasets["train"].features['cc_labels'].feature.names

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

    config = BertCausalmConfig(
        cc_heads_cfg=[CausalmHeadConfig(head_name='NER', head_type=TOKEN_CLASSIFICATION, head_params={'num_labels': len(label_list)})],
    )
    model = BertForCausalmAdditionalPreTraining(config)

    # noinspection PyTypeChecker
    args = TrainingArguments(
        f"test-product-ner-domain",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy='no',
    )
    args._n_gpu = 1

    data_collator = DataCollatorForCausalmAdditionalPretraining(tokenizer, collate_cc=True, mlm_probability=.15)

    metric = load_metric("seqeval")

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.evaluate()

    print(show_results())
    print()
