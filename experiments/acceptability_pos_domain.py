import json
import time
import warnings
from argparse import Namespace

import pandas
from datasets import Features, Value, Sequence, ClassLabel, DatasetDict, Dataset
from transformers import logging, BertModel, DataCollatorWithPadding

from experiments.pipelines import ate_estimation_pipeline, additional_pretraining_pipeline, classification_pipeline
from utils import DATA_DIR, tokenize_and_align_labels, PROJECT_DIR, BertTokenizerFast, BERT_MODEL_CHECKPOINT, SEQUENCE_CLASSIFICATION, \
    TOKEN_CLASSIFICATION, CausalMetrics

logger = logging.get_logger(__name__)


def get_acceptability_pos_domain_data(tokenizer, version='balanced'):
    # load raw data to memory
    train_df = pandas.read_json(str(DATA_DIR / 'acceptability_pos_domain' / f'acceptability_pos_domain_{version}_train.json'))
    test_df = pandas.read_json(str(DATA_DIR / 'acceptability_pos_domain' / f'acceptability_pos_domain_{version}_test.json'))
    train_df.index = train_df.index.rename('id')
    test_df.index = test_df.index.rename('id')

    test_df = test_df[test_df['tokens_cf'].notna()]

    # merge factual text into counterfactual text for examples w/o textual counterfactual
    def merge_f_cf(row, col_name):
        return row[f'{col_name}_cf'] if row[f'{col_name}_cf'] is not None else row[col_name]

    train_df['tokens_cf_merged'] = train_df.apply(axis=1, func=merge_f_cf, col_name='tokens')
    train_df['pos_tags_cf_merged'] = train_df.apply(axis=1, func=merge_f_cf, col_name='pos_tags')
    test_df['tokens_cf_merged'] = test_df.apply(axis=1, func=merge_f_cf, col_name='tokens')
    test_df['pos_tags_cf_merged'] = test_df.apply(axis=1, func=merge_f_cf, col_name='pos_tags')

    # get label names
    with open(str(DATA_DIR / 'label_names.json')) as f:
        label_names = json.load(f)

    # construct features
    features = Features({
        'id': Value(dtype='string', id='id'),
        'tokens': Sequence(Value(dtype='string'), id='tokens'),
        'tokens_cf': Sequence(Value(dtype='string'), id='tokens_cf'),
        'tokens_cf_merged': Sequence(Value(dtype='string'), id='tokens_cf_merged'),  # this column is here because not all texts have a cf.
        'pos_tags': Sequence(ClassLabel(num_classes=len(label_names['idx2pos']), names=label_names['idx2pos']), id='pos_tags'),
        'pos_tags_cf': Sequence(ClassLabel(num_classes=len(label_names['idx2pos']), names=label_names['idx2pos']), id='pos_tags_cf'),
        'pos_tags_cf_merged': Sequence(ClassLabel(num_classes=len(label_names['idx2pos']), names=label_names['idx2pos']), id='pos_tags_cf_merged'),
        'acceptability_amitavasil': ClassLabel(id='acceptability_amitavasil', num_classes=len(label_names['acceptability_names']),
                                               names=label_names['acceptability_names']),
        'sentiment': ClassLabel(id='sentiment', num_classes=len(label_names['sentiment_names']), names=label_names['sentiment_names']),
        'domain': ClassLabel(id='domain', num_classes=len(label_names['idx2domain']), names=label_names['idx2domain']),
        'review_text': Value(dtype='string', id='review_text')
    })

    # create HuggingFace DatasetDict
    datasets = DatasetDict()
    datasets['train'] = Dataset.from_pandas(train_df, features=features)
    datasets['test'] = Dataset.from_pandas(test_df, features=features)

    # rename columns to generic names
    datasets = datasets.rename_column('acceptability_amitavasil', 'tc_labels')
    datasets = datasets.rename_column('sentiment', 'task_labels')

    # tokenize and align labels
    datasets_f = datasets.map(tokenize_and_align_labels, batched=True,
                              fn_kwargs={'tokenizer': tokenizer, 'label_names': ['pos_tags'], 'tokens_key': 'tokens'})
    datasets_f = datasets_f.remove_columns(['tokens_cf', 'tokens_cf_merged', 'pos_tags_cf', 'pos_tags_cf_merged'])
    datasets_f = datasets_f.rename_column('pos_tags', 'cc_labels')

    datasets_cf = datasets.map(tokenize_and_align_labels, batched=True,
                               fn_kwargs={'tokenizer': tokenizer, 'label_names': ['pos_tags_cf_merged'], 'tokens_key': 'tokens_cf_merged'})
    datasets_cf = datasets_cf.remove_columns(['tokens', 'tokens_cf', 'pos_tags', 'pos_tags_cf'])
    datasets_cf = datasets_cf.rename_column('tokens_cf_merged', 'tokens')
    datasets_cf = datasets_cf.rename_column('pos_tags_cf_merged', 'cc_labels')

    return datasets_f, datasets_cf


def estimate_ate():
    ate_balanced = ate_estimation_pipeline(get_acceptability_pos_domain_data, version='balanced')
    ate_aggressive = ate_estimation_pipeline(get_acceptability_pos_domain_data, version='aggressive')
    ate_moogzam = ate_estimation_pipeline(get_acceptability_pos_domain_data, version='moogzam')
    print(f'Balanced:   {ate_balanced:.3f}')
    print(f'Aggressive: {ate_aggressive:.3f}')
    print(f'Moogzam:    {ate_moogzam:.3f}')


def causalm_pipeline(args):
    # initialize model
    time_str = time.strftime('%Y_%m_%d__%H_%M_%S')
    model_name = f'APD__{time_str}'
    save_dir = str(PROJECT_DIR / 'saved_models' / model_name)
    # logging.set_verbosity(logging.ERROR)
    warnings.filterwarnings('ignore', message='Was asked to gather along dimension 0, ')
    warnings.filterwarnings('ignore', message='\w* seems not to be NE tag.')

    # create tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_CHECKPOINT)

    # load data
    dataset_f, dataset_cf = get_acceptability_pos_domain_data(tokenizer, args.version)
    cc_label_list = dataset_f['train'].features['cc_labels'].feature.names
    tc_label_list = dataset_f['train'].features['tc_labels'].names
    task_label_list = dataset_f['train'].features['task_labels'].names

    # 1) Pretraining
    logger.info('**** 1) Pretraining ****')
    bert_o = BertModel.from_pretrained(BERT_MODEL_CHECKPOINT)

    # 2) Additional Pretraining
    logger.info('**** 2) Additional Pretraining ****')
    bert_cf = additional_pretraining_pipeline(tokenizer, dataset_f['train'], tc_head_type=SEQUENCE_CLASSIFICATION,
                                              cc_head_type=TOKEN_CLASSIFICATION, epochs=2)

    # 3) Downstream Task Training
    logger.info('**** 3) Downstream Task Training ****')
    bert_o_task_classifier, bert_o_task_classifier_metrics = classification_pipeline(tokenizer, bert_o, dataset_f, SEQUENCE_CLASSIFICATION, 'task',
                                                                                     task_label_list, epochs=5)
    bert_cf_task_classifier, bert_cf_task_classifier_metrics = classification_pipeline(tokenizer, bert_cf, dataset_f, SEQUENCE_CLASSIFICATION, 'task',
                                                                                       task_label_list, epochs=5)

    # 4a) Probing for Treated Concept
    logger.info('**** 4a) Probing for Treated Concept ****')
    _, bert_o_tc_classifier_metrics = classification_pipeline(tokenizer, bert_o, dataset_f, SEQUENCE_CLASSIFICATION, 'tc', tc_label_list, epochs=5)
    _, bert_cf_tc_classifier_metrics = classification_pipeline(tokenizer, bert_cf, dataset_f, SEQUENCE_CLASSIFICATION, 'tc', tc_label_list, epochs=5)

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
    conexp = metrics_cls.conexp(model=bert_o_task_classifier, dataset=dataset_f['test'], tc_indicator_name='tc_labels')

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
    results_df.to_csv(str(PROJECT_DIR / 'results' / f'{args.version}__{model_name}.csv'))


if __name__ == '__main__':
    # estimate_ate()
    causalm_pipeline(Namespace(version='balanced'))
    causalm_pipeline(Namespace(version='aggressive'))
    causalm_pipeline(Namespace(version='moogzam'))

