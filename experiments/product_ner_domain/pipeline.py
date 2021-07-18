import time
import warnings

import numpy as np
import pandas
from datasets import load_metric, Features, Value, ClassLabel, load_dataset, Sequence, DatasetDict, Dataset
from transformers import BertTokenizerFast, BertTokenizer, BertModel, TrainingArguments
from transformers import logging

from modeling.BERT.bert_causalm import BertForCausalmAdditionalPreTraining, BertCausalmForTokenClassification
from modeling.BERT.configuration_causalm import BertCausalmConfig, CausalmHeadConfig
from modeling.BERT.trainer_causalm import CausalmTrainingArguments, CausalmTrainer
from utils import DATA_DIR, BERT_MODEL_CHECKPOINT, PROJECT_DIR, CausalmMetrics, TOKEN_CLASSIFICATION, \
    DataCollatorForCausalmAdditionalPretraining, DataCollatorForCausalmTokenClassification


def tokenize_and_align_labels(examples, text_key='text', tokenizer=None, label_all_tokens=True, label_names=None):
    tokenized_inputs = tokenizer(examples[text_key], truncation=True, is_split_into_words=True)

    for label_name in label_names:
        labels = []
        for i, label in enumerate(examples[label_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs[label_name] = labels

    if 'task_labels' in examples:
        tokenized_inputs['task_labels'] = examples['task_labels']

    return tokenized_inputs


def additional_pretraining_pipeline(
        tokenizer,
        train_dataset,
        eval_dataset=None,
        epochs=5,
        save_dir=None
):
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

    return trainer


def downstream_task_pipeline(
        train_dataset,
        test_dataset,
        bert_model,
        token_classifier_type,
        num_labels,
        lr=2e-5,
        epochs=7,
        overfit=False,
        skip_training=False,
):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_CHECKPOINT)

    label_name = f'{token_classifier_type}_labels'

    config = BertCausalmConfig(num_labels=num_labels, token_classifier_type=token_classifier_type)
    model = BertCausalmForTokenClassification(config)

    model.bert = bert_model

    label_list = train_dataset.features[label_name].feature.names

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

    # noinspection PyTypeChecker
    args = TrainingArguments(
        output_dir=f'PND_{label_name}',
        evaluation_strategy='epoch',
        learning_rate=lr,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_strategy='no',
        label_names=[label_name],
    )

    labels_to_ignore = {'cc_labels'} if label_name == 'tc_labels' else {'tc_labels'} if label_name == 'cc_labels' else set()
    data_collator = DataCollatorForCausalmTokenClassification(tokenizer, label_name=label_name, labels_to_ignore=labels_to_ignore)

    metric = load_metric("seqeval")

    trainer = CausalmTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset if overfit else test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if skip_training:
        trainer.evaluate()
    else:
        trainer.train()

    return trainer


def get_ps_ner_domain_data(tokenizer, cf=False):
    train_df = pandas.read_pickle(str(DATA_DIR / 'PND_train.pkl'))
    test_df = pandas.read_pickle(str(DATA_DIR / 'PND_test.pkl'))

    domain_names = ['Books', 'Clothing', 'Electronics', 'Movies', 'Tools']
    ps_tags_names = ['not_ps'] + domain_names
    ner_names = ['NOT_ENTITY', 'CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE',
                 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT',
                 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']  # from https://spacy.io/models/en#en_core_web_trf-labels
    sentiment_names = ['negative', 'positive']

    features = Features({
        'review_text': Value(dtype='string', id='review_text'),
        'domain': ClassLabel(num_classes=len(domain_names), names=domain_names, id='domain'),
        'ps': Sequence(Value(dtype='string'), id='ps'),
        'sentiment': ClassLabel(num_classes=len(sentiment_names), names=sentiment_names, id='sentiment'),
        'tokens': Sequence(Value(dtype='string'), id='tokens'),
        'ps_tags': Sequence(ClassLabel(num_classes=len(ps_tags_names), names=ps_tags_names), id='ps_tags'),
        'ner_tags': Sequence(ClassLabel(num_classes=len(ner_names), names=ner_names), id='ner_tags'),
        'ps_cf': Sequence(Value(dtype='string'), id='ps_cf'),
    })

    datasets = DatasetDict()
    datasets['train'] = Dataset.from_pandas(train_df, features=features)
    datasets['test'] = Dataset.from_pandas(test_df, features=features)

    datasets = datasets.rename_column('ps_cf', 'tokens_cf')
    datasets = datasets.rename_column('ps_tags', 'tc_labels')
    datasets = datasets.rename_column('ner_tags', 'cc_labels')
    datasets = datasets.rename_column('sentiment', 'task_labels')

    labels_names = ['cc_labels', 'tc_labels']

    datasets = datasets.remove_columns(['ps', 'review_text'])

    key = 'tokens_cf' if cf else 'tokens'
    datasets = datasets.map(tokenize_and_align_labels, batched=True,
                            fn_kwargs={'tokenizer': tokenizer, 'label_names': labels_names, 'text_key': key})

    return datasets


def debug_main():
    time_str = time.strftime('%Y_%m_%d__%H_%M_%S')
    model_name = f'PND__{time_str}'
    save_dir = str(PROJECT_DIR / 'saved_models' / model_name)
    # logging.set_verbosity(logging.ERROR)
    warnings.filterwarnings('ignore', message='Was asked to gather along dimension 0, ')

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_CHECKPOINT)

    datasets = load_dataset("conll2003")
    datasets = datasets.rename_column('ner_tags', 'cc_labels')
    datasets = datasets.rename_column('pos_tags', 'tc_labels')
    cc_label_list = datasets["train"].features['cc_labels'].feature.names
    tc_label_list = datasets["train"].features['tc_labels'].feature.names

    print(f'num tc labels: {len(tc_label_list)}')
    print(f'num cc labels: {len(cc_label_list)}')

    datasets = datasets.map(tokenize_and_align_labels, batched=True,
                            fn_kwargs={'tokenizer': tokenizer, 'label_names': ['cc_labels', 'tc_labels']})

    print('>>> Additional pretraining')
    bert_o = BertModel.from_pretrained(BERT_MODEL_CHECKPOINT)
    bert_cf = additional_pretraining_pipeline(tokenizer, datasets['train'], save_dir=None, epochs=2).model.bert

    # probe forget treated concept --------------------
    print('>>> Treated concept training bert_o')
    bert_o_tc_classifier = downstream_task_pipeline(
        datasets['train'], datasets['test'], bert_o, epochs=3, token_classifier_type='tc', num_labels=len(tc_label_list)).model

    print('>>> Treated concept training bert_cf')
    bert_cf_tc_classifier = downstream_task_pipeline(
        datasets['train'], datasets['test'], bert_cf, epochs=3, token_classifier_type='tc', num_labels=len(tc_label_list)).model

    # probe remember control concept --------------------
    print('>>> Control concept training bert_o')
    bert_o_cc_classifier = downstream_task_pipeline(
        datasets['train'], datasets['test'], bert_o, epochs=3, token_classifier_type='cc', num_labels=len(cc_label_list)).model

    print('>>> Control concept training bert_cf')
    bert_cf_cc_classifier = downstream_task_pipeline(
        datasets['train'], datasets['test'], bert_cf, epochs=3, token_classifier_type='cc', num_labels=len(cc_label_list)).model

    # # train for task --------------------
    # print('>>> Downstream task training bert_o')
    # bert_o_task_classifier = downstream_task_training_pipeline(
    #     datasets['train'], datasets['test'], bert_o, epochs=7, label_names=['task_labels']).model
    #
    # print('>>> Downstream task training bert_cf')
    # bert_cf_task_classifier = downstream_task_training_pipeline(
    #     datasets['train'], datasets['test'], bert_cf, epochs=7, label_names=['task_labels']).model

    # print('>>> Computing metrics')
    # # metrics
    # metrics_cls = CausalmMetrics(BERT_MODEL_CHECKPOINT)
    # conexp = metrics_cls.conexp(model=bert_o_task_classifier, dataset=dataset_f['test'])
    # treate = metrics_cls.treate(model_o=bert_o_task_classifier, model_cf=bert_cf_task_classifier, dataset=dataset_f['test'])
    # ate = metrics_cls.ate(model=bert_o_task_classifier, dataset_f=dataset_f['test'], dataset_cf=dataset_cf['test'])
    #
    # print('\n\n\n\n')
    # print(f'CONEXP: {conexp:.3f}')
    # print(f'TReATE: {treate:.3f}')
    # print(f'ATE_gt: {ate:.3f}')


def main():
    # initialize model
    time_str = time.strftime('%Y_%m_%d__%H_%M_%S')
    model_name = f'PND__{time_str}'
    save_dir = str(PROJECT_DIR / 'saved_models' / model_name)
    logging.set_verbosity(logging.ERROR)
    warnings.filterwarnings('ignore', message='Was asked to gather along dimension 0, ')

    # create tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_CHECKPOINT)

    # load data
    dataset_f = get_ps_ner_domain_data(tokenizer, cf=False)
    dataset_cf = get_ps_ner_domain_data(tokenizer, cf=True)
    cc_label_list = dataset_f['train'].features['cc_labels'].feature.names
    tc_label_list = dataset_f['train'].features['tc_labels'].feature.names
    task_label_list = dataset_f['train'].features['task_labels'].feature.names

    # 1) Pretraining
    bert_o = BertModel.from_pretrained(BERT_MODEL_CHECKPOINT)

    # 2) Additional Pretraining
    bert_cf = additional_pretraining_pipeline(tokenizer, dataset_f['train'], epochs=2).model.bert

    # 3) Downstream Task Training
    bert_o_task_classifier = downstream_task_pipeline(dataset_f['train'], dataset_f['test'], bert_o, 'task', num_labels=len(task_label_list), epochs=7).model
    bert_cf_task_classifier = downstream_task_pipeline(dataset_f['train'], dataset_f['test'], bert_cf, 'task', num_labels=len(task_label_list), epochs=7).model

    # 4) Probing
    # Probing for Treated Concept
    bert_o_tc_classifier = downstream_task_pipeline(dataset_f['train'], dataset_f['test'], bert_o, 'tc', num_labels=len(tc_label_list), epochs=7).model
    bert_cf_tc_classifier = downstream_task_pipeline(dataset_f['train'], dataset_f['test'], bert_cf, 'tc', num_labels=len(tc_label_list), epochs=7).model

    # Probing for Control Concept
    bert_o_cc_classifier = downstream_task_pipeline(dataset_f['train'], dataset_f['test'], bert_o, 'cc', num_labels=len(cc_label_list), epochs=7).model
    bert_cf_cc_classifier = downstream_task_pipeline(dataset_f['train'], dataset_f['test'], bert_cf, 'cc', num_labels=len(cc_label_list), epochs=7).model

    # 5) ATEs Estimation
    metrics_cls = CausalmMetrics(BERT_MODEL_CHECKPOINT)
    treate = metrics_cls.treate(model_o=bert_o_task_classifier, model_cf=bert_cf_task_classifier, dataset=dataset_f['test'])
    ate = metrics_cls.ate(model=bert_o_task_classifier, dataset_f=dataset_f['test'], dataset_cf=dataset_cf['test'])
    conexp = metrics_cls.conexp(model=bert_o_task_classifier, dataset=dataset_f['test'])

    # save results
    results_df = pandas.DataFrame.from_records({
        'Task': ['Sentiment'],
        'Adversarial Task': ['Product Specific'],
        'Control Task': ['NER'],
        'Confounder': ['Domain'],

        'BERT-O Treated Performance': [],
        'BERT-CF Treated Performance': [],
        'INLP Treated Performance': [None],

        'BERT-O Control Performance': [],
        'BERT-CF Control Performance': [],
        'INLP Control Performance': [None],

        'BERT-O Task Performance': [],
        'BERT-CF Task Performance': [],
        'INLP Task Performance': [None],

        'Treated ATE': [ate],
        'Treated TReATE': [treate],
        'Treated INLP': [None],
        'Treated CONEXP': [conexp],
    })
    print('\n' * 3)
    print(results_df)
    results_df.to_csv(str(PROJECT_DIR / 'results' / model_name))


if __name__ == '__main__':
    main()
