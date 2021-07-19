import time
import warnings

import numpy as np
import pandas
from datasets import load_metric
from transformers import BertTokenizerFast, BertTokenizer, BertModel, TrainingArguments, DataCollatorWithPadding
from transformers import logging

from experiments.product_ner_domain.dataset import get_ps_ner_domain_data
from modeling.BERT.bert_causalm import BertForCausalmAdditionalPreTraining, BertCausalmForTokenClassification, BertCausalmForSequenceClassification
from modeling.BERT.configuration_causalm import BertCausalmConfig, CausalmHeadConfig
from modeling.BERT.trainer_causalm import CausalmTrainingArguments, CausalmTrainer
from utils import BERT_MODEL_CHECKPOINT, PROJECT_DIR, CausalMetrics, TOKEN_CLASSIFICATION, \
    DataCollatorForCausalmAdditionalPretraining, DataCollatorForCausalmTokenClassification, EvalMetrics, SEQUENCE_CLASSIFICATION

logger = logging.get_logger(__name__)


def additional_pretraining_pipeline(tokenizer, train_dataset, eval_dataset=None, epochs=5, save_dir=None):
    data_collator = DataCollatorForCausalmAdditionalPretraining(tokenizer=tokenizer, mlm_probability=0.15, collate_cc=True, collate_tc=True)

    # model
    num_tc_labels = train_dataset.features['tc_labels'].feature.num_classes
    num_cc_labels = train_dataset.features['cc_labels'].feature.num_classes
    config = BertCausalmConfig(
        tc_heads_cfg=[CausalmHeadConfig(head_name='tc', head_type=TOKEN_CLASSIFICATION, head_params={'num_labels': num_tc_labels})],
        cc_heads_cfg=[CausalmHeadConfig(head_name='cc', head_type=TOKEN_CLASSIFICATION, head_params={'num_labels': num_cc_labels})],
        tc_lambda=0.2,
    )
    model = BertForCausalmAdditionalPreTraining(config)

    # training
    # noinspection PyTypeChecker
    args = CausalmTrainingArguments(
        'product_ner_domain',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_strategy='no',
        label_names=['cc_labels', 'tc_labels'],
        causalm_additional_pretraining=True,
        num_cc=1,
        num_tc=1
    )

    # uncomment this to cancel parallel training
    args._n_gpu = 1

    # noinspection PyTypeChecker
    trainer = CausalmTrainer(
        model,
        args,
        eval_dataset=train_dataset if not eval_dataset else eval_dataset,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    if save_dir:
        model.bert.save_pretrained(save_directory=save_dir)
        print(f'>>> Saved model.bert in {save_dir}')

    return trainer.model.bert


def classification_pipeline(tokenizer, bert_model, dataset, task_type, classifier_type, labels_list, epochs=5, parallel=True):
    """
    Classification training pipeline.

    task_type: {TOKEN_CLASSIFICATION, SEQ_CLASSIFICATION}
    classifier_type: {'tc', 'cc', 'task'}
    """
    # initialize model
    if task_type == SEQUENCE_CLASSIFICATION:
        config = BertCausalmConfig(num_labels=len(labels_list), sequence_classifier_type=classifier_type)
        model = BertCausalmForSequenceClassification(config)
    elif task_type == TOKEN_CLASSIFICATION:
        config = BertCausalmConfig(num_labels=len(labels_list), token_classifier_type=classifier_type)
        model = BertCausalmForTokenClassification(config)
    else:
        raise NotImplementedError(f'Unsupported task type "{task_type}"')
    model.bert = bert_model

    # initialize metrics
    metrics_cls = EvalMetrics(labels_list)
    if task_type == SEQUENCE_CLASSIFICATION:
        compute_metrics = metrics_cls.compute_sequence_classification_f1
    elif task_type == TOKEN_CLASSIFICATION:
        compute_metrics = metrics_cls.compute_token_classification_f1
    else:
        raise NotImplementedError(f'Unsupported task type "{task_type}"')

    # initialize labels
    label_name = f'{classifier_type}_labels'
    labels_to_ignore = {'cc_labels', 'tc_labels', 'task_labels'}
    labels_to_ignore.remove(label_name)
    dataset = dataset.remove_columns(list(labels_to_ignore))

    # initialize data collator
    if task_type == SEQUENCE_CLASSIFICATION:
        data_collator = None
    elif task_type == TOKEN_CLASSIFICATION:
        data_collator = DataCollatorForCausalmTokenClassification(tokenizer, label_name, labels_to_ignore)
    else:
        raise NotImplementedError(f'Unsupported task type "{task_type}"')

    # noinspection PyTypeChecker
    args = TrainingArguments(
        output_dir=f'PND_{label_name}',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_strategy='no',
        label_names=[label_name],
    )

    if not parallel:
        args._n_gpu = 1

    # initialize trainer
    trainer = CausalmTrainer(
        model,
        args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()

    return trainer.model, eval_results


def main():
    # initialize model
    time_str = time.strftime('%Y_%m_%d__%H_%M_%S')
    model_name = f'PND__{time_str}'
    save_dir = str(PROJECT_DIR / 'saved_models' / model_name)
    # logging.set_verbosity(logging.ERROR)
    warnings.filterwarnings('ignore', message='Was asked to gather along dimension 0, ')
    warnings.filterwarnings('ignore', message='\w* seems not to be NE tag.')

    # create tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_CHECKPOINT)

    # load data
    dataset_f, dataset_cf = get_ps_ner_domain_data(tokenizer)
    cc_label_list = dataset_f['train'].features['cc_labels'].feature.names
    tc_label_list = dataset_f['train'].features['tc_labels'].feature.names
    task_label_list = dataset_f['train'].features['task_labels'].names

    # debug
    DEBUG_EPOCHS = 1

    # 1) Pretraining
    logger.info('**** 1) Pretraining ****')
    bert_o = BertModel.from_pretrained(BERT_MODEL_CHECKPOINT)

    # 2) Additional Pretraining
    logger.info('**** 2) Additional Pretraining ****')
    bert_cf = additional_pretraining_pipeline(tokenizer, dataset_f['train'], epochs=2)

    # 3) Downstream Task Training
    logger.info('**** 3) Downstream Task Training ****')
    bert_o_task_classifier, bert_o_task_classifier_metrics = classification_pipeline(tokenizer, bert_o, dataset_f, SEQUENCE_CLASSIFICATION, 'task',
                                                                                     task_label_list, epochs=5)
    bert_cf_task_classifier, bert_cf_task_classifier_metrics = classification_pipeline(tokenizer, bert_cf, dataset_f, SEQUENCE_CLASSIFICATION, 'task',
                                                                                       task_label_list, epochs=5)

    # 4a) Probing for Treated Concept
    logger.info('**** 4a) Probing for Treated Concept ****')
    _, bert_o_tc_classifier_metrics = classification_pipeline(tokenizer, bert_o, dataset_f, TOKEN_CLASSIFICATION, 'tc', tc_label_list, epochs=5)
    _, bert_cf_tc_classifier_metrics = classification_pipeline(tokenizer, bert_cf, dataset_f, TOKEN_CLASSIFICATION, 'tc', tc_label_list, epochs=5)

    # 4b) Probing for Control Concept
    logger.info('**** 4b) Probing for Control Concept ****')
    _, bert_o_cc_classifier_metrics = classification_pipeline(tokenizer, bert_o, dataset_f, TOKEN_CLASSIFICATION, 'cc', cc_label_list, epochs=5)
    _, bert_cf_cc_classifier_metrics = classification_pipeline(tokenizer, bert_cf, dataset_f, TOKEN_CLASSIFICATION, 'cc', cc_label_list, epochs=5)

    # 5) ATEs Estimation
    logger.info('**** 5) ATEs Estimation ****')
    data_collator = DataCollatorWithPadding(tokenizer)  # for sentiment classification
    metrics_cls = CausalMetrics(data_collator)
    treate = metrics_cls.treate(model_o=bert_o_task_classifier, model_cf=bert_cf_task_classifier, dataset=dataset_f['test'])
    ate = metrics_cls.ate(model=bert_o_task_classifier, dataset_f=dataset_f['test'], dataset_cf=dataset_cf['test'])
    conexp = metrics_cls.conexp(model=bert_o_task_classifier, dataset=dataset_f['test'], tc_indicator_name='is_ps')

    # save results
    results_df = pandas.DataFrame.from_records({
        'Task': ['Sentiment'],
        'Adversarial Task': ['Product Specific'],
        'Control Task': ['NER'],
        'Confounder': ['Domain'],

        'BERT-O Treated Performance': [bert_o_tc_classifier_metrics['eval_f1']],
        'BERT-CF Treated Performance': [bert_cf_tc_classifier_metrics['eval_f1']],
        'INLP Treated Performance': [None],

        'BERT-O Control Performance': [bert_o_cc_classifier_metrics['eval_f1']],
        'BERT-CF Control Performance': [bert_cf_cc_classifier_metrics['eval_f1']],
        'INLP Control Performance': [None],

        'BERT-O Task Performance': [bert_o_task_classifier_metrics['eval_f1']],
        'BERT-CF Task Performance': [bert_cf_task_classifier_metrics['eval_f1']],
        'INLP Task Performance': [None],

        'Treated ATE': [ate],
        'Treated TReATE': [treate],
        'Treated INLP': [None],
        'Treated CONEXP': [conexp],
    })
    results_df.to_csv(str(PROJECT_DIR / 'results' / f'{model_name}.csv'))


if __name__ == '__main__':
    main()
