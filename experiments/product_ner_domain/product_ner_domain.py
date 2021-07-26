import time
import warnings

import pandas
from datasets import Features, Value, ClassLabel, Sequence, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertModel, DataCollatorWithPadding
from transformers import logging

from experiments.pipelines import additional_pretraining_pipeline, classification_pipeline, ate_estimation_pipeline
from utils import BERT_MODEL_CHECKPOINT, PROJECT_DIR, CausalMetrics, TOKEN_CLASSIFICATION, \
    SEQUENCE_CLASSIFICATION, DATA_DIR, tokenize_and_align_labels, get_label_names

logger = logging.get_logger(__name__)


def get_ps_ner_domain_data(tokenizer, version='balanced'):
    # load raw data
    if version == 'original':
        train_df = pandas.read_pickle(str(DATA_DIR / 'PND_train.pkl'))
        test_df = pandas.read_pickle(str(DATA_DIR / 'PND_test.pkl'))
    else:
        df = pandas.read_json(str(DATA_DIR / 'product_ner_domain' / f'product_ner_domain_{version}.json'))
        df = df.drop(columns=['ps_tags_noman_cf', 'ner_tags_spacy_cf'])
        df.index = df.index.rename('id')
        train_df, test_df = train_test_split(df, test_size=0.25)

    label_names = get_label_names()

    datasets = DatasetDict()
    datasets['train'] = Dataset.from_pandas(train_df)
    datasets['test'] = Dataset.from_pandas(test_df)

    # construct features
    features = Features({
        'id': Value(dtype='string', id='id'),
        'tokens': Sequence(Value(dtype='string'), id='tokens'),
        'tokens_cf': Sequence(Value(dtype='string'), id='tokens_cf'),
        'ps_tags_noman': Sequence(ClassLabel(num_classes=len(label_names['ps_tags_names']), names=label_names['ps_tags_names']), id='ps_tags_noman'),
        # 'ps_tags_noman_cf': Sequence(ClassLabel(num_classes=len(label_names['ps_tags_names']), names=label_names['ps_tags_names']), id='ps_tags_noman_cf'),
        'ner_tags_spacy': Sequence(ClassLabel(num_classes=len(label_names['idx2ne']), names=label_names['idx2ne']), id='ner_tags_spacy'),
        # 'ner_tags_spacy_cf': Sequence(ClassLabel(num_classes=len(label_names['idx2ne']), names=label_names['idx2ne']), id='ner_tags_spacy_cf'),
        'sentiment': ClassLabel(num_classes=len(label_names['sentiment_names']), names=label_names['sentiment_names'], id='sentiment'),
        'domain': ClassLabel(num_classes=len(label_names['idx2domain']), names=label_names['idx2domain'], id='domain'),
    })

    # create HuggingFace DatasetDict
    datasets = DatasetDict()
    datasets['train'] = Dataset.from_pandas(train_df, features=features)
    datasets['test'] = Dataset.from_pandas(test_df, features=features)

    # rename columns to generic names
    datasets = datasets.rename_column('ps_tags_noman', 'tc_labels')
    datasets = datasets.rename_column('ner_tags_spacy', 'cc_labels')
    datasets = datasets.rename_column('sentiment', 'task_labels')

    # tokenize and align labels
    labels_names = ['cc_labels', 'tc_labels']
    datasets_f = datasets.map(tokenize_and_align_labels, batched=True,
                              fn_kwargs={'tokenizer': tokenizer, 'label_names': labels_names, 'tokens_key': 'tokens'})
    datasets_f = datasets_f.remove_columns(['tokens_cf'])

    datasets_cf = datasets.map(tokenize_and_align_labels, batched=True,
                               fn_kwargs={'tokenizer': tokenizer, 'label_names': labels_names, 'tokens_key': 'tokens_cf'})
    datasets_cf = datasets_cf.remove_columns(['tokens'])
    datasets_cf = datasets_cf.rename_column('tokens_cf', 'tokens')

    return datasets_f, datasets_cf


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


def estimate_ate():
    ate_balanced = ate_estimation_pipeline(get_ps_ner_domain_data, version='balanced')
    ate_aggressive = ate_estimation_pipeline(get_ps_ner_domain_data, version='aggressive')
    ate_moogzam = ate_estimation_pipeline(get_ps_ner_domain_data, version='moogzam')
    print(f'Balanced:   {ate_balanced:.3f}')
    print(f'Aggressive: {ate_aggressive:.3f}')
    print(f'Moogzam:    {ate_moogzam:.3f}')


if __name__ == '__main__':
    estimate_ate()
