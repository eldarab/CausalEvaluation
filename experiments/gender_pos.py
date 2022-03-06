import time
import warnings
from argparse import Namespace

from datasets import Features, DatasetDict, Dataset, Value, Sequence, ClassLabel
from transformers import BertModel, DataCollatorWithPadding
from transformers.utils import logging

from experiments.pipelines import ate_estimation_pipeline, additional_pretraining_pipeline, classification_pipeline
from utils import DATA_DIR, json, tokenize_and_align_labels, PROJECT_DIR, BertTokenizerFast, BERT_MODEL_CHECKPOINT, TOKEN_CLASSIFICATION, \
    CausalMetrics, SEQUENCE_CLASSIFICATION
import pandas as pd

logger = logging.get_logger(__name__)


def get_gender_pos_data(tokenizer, version='balanced'):
    # load raw data to memory
    train_df = pd.read_json(str(DATA_DIR / 'gender_pos' / f'gender_pos_train_{version}.json'))
    test_df = pd.read_json(str(DATA_DIR / 'gender_pos' / f'gender_pos_test_{version}.json'))
    train_df.index = train_df.index.rename('id')
    test_df.index = test_df.index.rename('id')

    # get label names
    with open(str(DATA_DIR / 'label_names.json')) as f:
        label_names = json.load(f)

    # construct features
    features = Features({
        'id': Value(dtype='string', id='id'),
        'review_text': Value(dtype='string', id='review_text'),
        'tokens': Sequence(Value(dtype='string'), id='tokens'),
        'tokens_cf': Sequence(Value(dtype='string'), id='tokens_cf'),
        'gender_specific_tags': Sequence(ClassLabel(num_classes=len(label_names['idx2gender']), names=label_names['idx2gender']),
                                         id='gender_specific_tags'),
        'pos_tags_spacy': Sequence(ClassLabel(num_classes=len(label_names['idx2pos']), names=label_names['idx2pos']), id='pos_tags'),
        'sentiment': ClassLabel(id='sentiment', num_classes=len(label_names['sentiment_names']), names=label_names['sentiment_names']),
        'domain': ClassLabel(id='domain', num_classes=len(label_names['idx2domain']), names=label_names['idx2domain']),
        'is_male_specific': ClassLabel(id='is_male_specific', num_classes=2)
    })

    # create HuggingFace DatasetDict
    datasets = DatasetDict()
    datasets['train'] = Dataset.from_pandas(train_df, features=features)
    datasets['test'] = Dataset.from_pandas(test_df, features=features)

    # rename columns to generic names
    datasets = datasets.rename_column('gender_specific_tags', 'tc_labels')
    datasets = datasets.rename_column('pos_tags_spacy', 'cc_labels')
    datasets = datasets.rename_column('sentiment', 'task_labels')

    # tokenize and align labels
    datasets_f = datasets.map(tokenize_and_align_labels, batched=True,
                              fn_kwargs={'tokenizer': tokenizer, 'label_names': ['tc_labels', 'cc_labels'], 'tokens_key': 'tokens'})
    datasets_f = datasets_f.remove_columns(['tokens_cf'])

    datasets_cf = datasets.map(tokenize_and_align_labels, batched=True,
                               fn_kwargs={'tokenizer': tokenizer, 'label_names': ['tc_labels', 'cc_labels'], 'tokens_key': 'tokens_cf'})
    datasets_cf = datasets_cf.remove_columns(['tokens'])
    datasets_cf = datasets_cf.rename_column('tokens_cf', 'tokens')

    return datasets_f, datasets_cf


def estimate_ate():
    ate_balanced = ate_estimation_pipeline(get_gender_pos_data, version='balanced')
    ate_aggressive = ate_estimation_pipeline(get_gender_pos_data, version='aggressive')
    ate_moogzam = ate_estimation_pipeline(get_gender_pos_data, version='extreme')
    print(f'Balanced:   {ate_balanced:.3f}')
    print(f'Aggressive: {ate_aggressive:.3f}')
    print(f'Extreme:    {ate_moogzam:.3f}')


def causalm_pipeline(args):
    # initialize model
    time_str = time.strftime('%Y_%m_%d__%H_%M_%S')
    model_name = f'GP__{time_str}'
    save_dir = str(PROJECT_DIR / 'saved_models' / model_name)
    # logging.set_verbosity(logging.ERROR)
    warnings.filterwarnings('ignore', message='Was asked to gather along dimension 0, ')
    warnings.filterwarnings('ignore', message='\w* seems not to be NE tag.')

    # create tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_CHECKPOINT)

    # load data
    dataset_f, dataset_cf = get_gender_pos_data(tokenizer, args.version)
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
    bert_cf = additional_pretraining_pipeline(tokenizer, dataset_f['train'], tc_head_type=TOKEN_CLASSIFICATION,
                                              cc_head_type=TOKEN_CLASSIFICATION, epochs=2)

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
    conexp = metrics_cls.conexp(model=bert_o_task_classifier, dataset=dataset_f['test'], tc_indicator_name='is_male_specific')

    # save results
    results_df = pd.DataFrame.from_records({
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
    causalm_pipeline(Namespace(version='extreme'))
