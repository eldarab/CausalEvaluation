import pandas
from datasets import Features, Value, ClassLabel, Sequence, DatasetDict, Dataset

from utils import DATA_DIR, tokenize_and_align_labels


def get_ps_ner_domain_data(tokenizer):
    # load raw data
    train_df = pandas.read_pickle(str(DATA_DIR / 'PND_train.pkl'))
    test_df = pandas.read_pickle(str(DATA_DIR / 'PND_test.pkl'))

    # get label names
    domain_names = ['Books', 'Clothing', 'Electronics', 'Movies', 'Tools']
    ps_tags_names = ['not_ps'] + domain_names
    ner_names = ['NOT_ENTITY', 'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE',
                 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT',
                 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']  # from https://spacy.io/models/en#en_core_web_trf-labels
    sentiment_names = ['negative', 'positive']

    # construct features
    features = Features({
        'review_text': Value(dtype='string', id='review_text'),
        'domain': ClassLabel(num_classes=len(domain_names), names=domain_names, id='domain'),
        'ps': Sequence(Value(dtype='string'), id='ps'),
        'sentiment': ClassLabel(num_classes=len(sentiment_names), names=sentiment_names, id='sentiment'),
        'tokens': Sequence(Value(dtype='string'), id='tokens'),
        'ps_tags': Sequence(ClassLabel(num_classes=len(ps_tags_names), names=ps_tags_names), id='ps_tags'),
        'ner_tags': Sequence(ClassLabel(num_classes=len(ner_names), names=ner_names), id='ner_tags'),
        'tokens_cf': Sequence(Value(dtype='string'), id='tokens_cf'),
        'is_ps': Value(dtype='int8', id='is_ps'),
        'num_masks': Value(dtype='int16', id='num_masks')
    })

    # create HuggingFace DatasetDict
    datasets = DatasetDict()
    datasets['train'] = Dataset.from_pandas(train_df, features=features)
    datasets['test'] = Dataset.from_pandas(test_df, features=features)

    # remove redundant columns
    datasets = datasets.remove_columns(['ps', 'review_text', 'num_masks'])

    # rename columns to generic names
    datasets = datasets.rename_column('ps_tags', 'tc_labels')
    datasets = datasets.rename_column('ner_tags', 'cc_labels')
    datasets = datasets.rename_column('sentiment', 'task_labels')

    # tokenize and align labels
    labels_names = ['cc_labels', 'tc_labels']

    datasets_f = datasets.map(tokenize_and_align_labels, batched=True,
                              fn_kwargs={'tokenizer': tokenizer, 'label_names': labels_names, 'text_key': 'tokens'})
    datasets_f = datasets_f.remove_columns(['tokens_cf'])

    datasets_cf = datasets.map(tokenize_and_align_labels, batched=True,
                               fn_kwargs={'tokenizer': tokenizer, 'label_names': labels_names, 'text_key': 'tokens_cf'})
    datasets_cf = datasets_cf.remove_columns(['tokens'])
    datasets_cf = datasets_cf.rename_column('tokens_cf', 'tokens')

    return datasets_f, datasets_cf
