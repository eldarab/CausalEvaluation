import time
import warnings

import numpy as np
from datasets import load_metric, Features, Value, ClassLabel, load_dataset
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling, BertTokenizer, BertModel, TrainingArguments, Trainer
from transformers import logging

from modeling.BERT.bert_causalm import BertForCausalmAdditionalPreTraining, BertCausalmForSequenceClassification
from modeling.BERT.configuration_causalm import BertCausalmConfig, CausalmHeadConfig
from modeling.BERT.trainer_causalm import CausalmTrainingArguments, CausalmTrainer
from utils import DATA_DIR, BERT_MODEL_CHECKPOINT, SEQUENCE_CLASSIFICATION, PROJECT_DIR, CausalmMetrics


def additional_pretraining_pipeline(
        tokenizer,
        train_dataset,
        eval_dataset=None,
        epochs=5,
        save_dir=None
):
    lm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # model
    config = BertCausalmConfig(
        tc_heads_cfg=[CausalmHeadConfig(head_name='acceptability', head_type=SEQUENCE_CLASSIFICATION, head_params={'num_labels': 2})],
        cc_heads_cfg=[CausalmHeadConfig(head_name='is_books', head_type=SEQUENCE_CLASSIFICATION, head_params={'num_labels': 2})],
        tc_lambda=0.2,
    )
    model = BertForCausalmAdditionalPreTraining(config)

    # training
    # noinspection PyTypeChecker
    args = CausalmTrainingArguments(
        'sanity_check',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_strategy='no',
        label_names=['cc_label', 'tc_label'],
        causalm_additional_pretraining=True,
        num_cc=1,
        num_tc=1
    )

    # noinspection PyTypeChecker
    trainer = CausalmTrainer(
        model,
        args,
        eval_dataset=train_dataset if not eval_dataset else eval_dataset,
        train_dataset=train_dataset,
        data_collator=lm_data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    if save_dir:
        model.bert.save_pretrained(save_directory=save_dir)
        print(f'>>> Saved model.bert in {save_dir}')

    return trainer


def downstream_task_training_pipeline(
        train_dataset,
        test_dataset,
        bert_model,
        num_labels=2,
        metric_name='accuracy',
        lr=2e-5,
        epochs=7,
        overfit=False,
        skip_training=False,
):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_CHECKPOINT)

    config = BertCausalmConfig(num_labels=num_labels, sequence_classifier_type='task')
    model = BertCausalmForSequenceClassification(config)

    model.bert = bert_model

    metric = load_metric(metric_name)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labels)

    # noinspection PyTypeChecker
    args = TrainingArguments(
        output_dir='sanity_check',
        evaluation_strategy='epoch',
        learning_rate=lr,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        save_strategy='no',
        label_names=['task_label'],
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset if overfit else test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if skip_training:
        trainer.evaluate()
    else:
        trainer.train()

    return trainer


def get_data(tokenizer):
    features = Features({
        'text': Value(dtype='string', id='text'),
        'acceptability_cf_amitavasil': Value(dtype='string', id='text_cf'),
        'acceptability_sophiemarshall2': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], id='tc_label'),
        'is_books': ClassLabel(num_classes=2, names=['not_books', 'books'], id='cc_label'),
        'sentiment': ClassLabel(num_classes=2, names=['negative', 'positive'], id='task_label'),
    })
    dataset_f = load_dataset(
        'csv',
        data_files={'train': str(DATA_DIR / 'SAD_train.csv'),
                    'test': str(DATA_DIR / 'SAD_test.csv')},
        index_col=0,
        features=features
    )
    dataset_f = dataset_f.rename_column('acceptability_sophiemarshall2', 'tc_label')
    dataset_f = dataset_f.rename_column('is_books', 'cc_label')
    dataset_f = dataset_f.rename_column('sentiment', 'task_label')

    dataset_cf = dataset_f.remove_columns(['text'])
    dataset_cf = dataset_cf.rename_column('acceptability_cf_amitavasil', 'text')

    dataset_f = dataset_f.remove_columns(['acceptability_cf_amitavasil'])

    def preprocess_function(examples):
        if not any(examples['text']):  # don't preprocess "None"
            return tokenizer(['' for _ in examples['text']], truncation=True, padding=True)
        else:
            return tokenizer(examples['text'], truncation=True, padding=True)

    dataset_f = dataset_f.map(preprocess_function, batched=True)
    dataset_cf = dataset_cf.map(preprocess_function, batched=True)

    return dataset_f, dataset_cf


def main():
    time_str = time.strftime('%Y_%m_%d__%H_%M_%S')
    model_name = f'SAD__{time_str}'
    save_dir = str(PROJECT_DIR / 'saved_models' / model_name)
    logging.set_verbosity(logging.ERROR)
    warnings.filterwarnings('ignore', message='Was asked to gather along dimension 0, ')

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_CHECKPOINT)

    dataset_f, dataset_cf = get_data(tokenizer)

    print('>>> Additional pretraining')
    bert_o = BertModel.from_pretrained(BERT_MODEL_CHECKPOINT)
    bert_cf = additional_pretraining_pipeline(tokenizer, dataset_f['train'], save_dir=None, epochs=5).model.bert

    print('>>> Downstream task training bert_o')
    bert_o_classifier = downstream_task_training_pipeline(dataset_f['train'], dataset_f['test'], bert_o, epochs=7).model
    print('>>> Downstream task training bert_cf')
    bert_cf_classifier = downstream_task_training_pipeline(dataset_f['train'], dataset_f['test'], bert_cf, epochs=11).model

    print('>>> Computing metrics')
    # metrics
    metrics_cls = CausalmMetrics(BERT_MODEL_CHECKPOINT)
    conexp = metrics_cls.conexp(model=bert_o_classifier, dataset=dataset_f['test'])
    treate = metrics_cls.treate(model_o=bert_o_classifier, model_cf=bert_cf_classifier, dataset=dataset_f['test'])
    ate = metrics_cls.ate(model=bert_o_classifier, dataset_f=dataset_f['test'], dataset_cf=dataset_cf['test'])

    print('\n\n\n\n')
    print(f'CONEXP: {conexp:.3f}')
    print(f'TReATE: {treate:.3f}')
    print(f'ATE_gt: {ate:.3f}')


if __name__ == '__main__':
    main()
