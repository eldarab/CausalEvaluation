import json

import pandas
from datasets import Features, Value, Sequence, ClassLabel, DatasetDict, Dataset
from sklearn.model_selection import train_test_split

from experiments.pipelines import ate_estimation_pipeline
from utils import DATA_DIR, tokenize_and_align_labels


def get_acceptability_pos_domain_data(tokenizer, version='balanced'):
    # load raw data to memory
    if version == 'balanced':
        df = pandas.read_pickle(str(DATA_DIR / 'acceptability_pos_domain' / 'acceptability_pos_domain.pkl'))
        train_df, test_df = train_test_split(df, test_size=0.25)
    elif version == 'aggressive':
        train_df = pandas.read_pickle(str(DATA_DIR / 'acceptability_pos_domain' / 'APD_aggressive_train.pkl'))
        test_df = pandas.read_pickle(str(DATA_DIR / 'acceptability_pos_domain' / 'APD_aggressive_test.pkl'))
    elif version == 'moogzam':
        train_df = pandas.read_pickle(str(DATA_DIR / 'acceptability_pos_domain' / 'APD_moogzam_train.pkl'))
        test_df = pandas.read_pickle(str(DATA_DIR / 'acceptability_pos_domain' / 'APD_moogzam_test.pkl'))
    else:
        raise RuntimeError(f'Illegal correlation "{version}"')

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


def main():
    pass


if __name__ == '__main__':
    estimate_ate()
