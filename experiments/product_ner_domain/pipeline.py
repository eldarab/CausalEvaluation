import time
import warnings

import numpy as np
import pandas
from datasets import load_metric, Features, Value, ClassLabel, Sequence, DatasetDict, Dataset
from transformers import BertTokenizerFast, BertTokenizer, BertModel, TrainingArguments
from transformers import logging


from modeling.BERT.bert_causalm import BertForCausalmAdditionalPreTraining, BertCausalmForTokenClassification, BertCausalmForSequenceClassification
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

    # uncomment this to cancel parallel training
    # args._n_gpu = 1

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


def classification_pipeline(tokenizer, bert_model, dataset, classifier_type, classification_task_type, label_name, labels_list, epochs=7):
    """
    Downstream task training pipeline.

    classification_task_type: {'text-classification', 'token-classification'}
    classifier_type: {'tc', 'cc', 'task'}
    """
    # initialize model
    if classification_task_type == 'text-classification':
        config = BertCausalmConfig(num_labels=len(labels_list), sequence_classifier_type=classifier_type)
        model = BertCausalmForSequenceClassification(config)
    elif classification_task_type == 'token-classification':
        config = BertCausalmConfig(num_labels=len(labels_list), token_classifier_type=classifier_type)
        model = BertCausalmForTokenClassification(config)
    else:
        raise NotImplementedError()
    model.bert = bert_model

    # initialize metrics
    metric = load_metric('accuracy')


    # initialize training
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



def downstream_task_pipeline(
        dataset,
        bert_model,
        classifier_type,
        label_list,
        lr=2e-5,
        epochs=7,
        overfit=False,
        skip_training=False,
):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_CHECKPOINT)

    label_name = f'{classifier_type}_labels'

    if classifier_type == 'tc' or classifier_type == 'cc':
        config = BertCausalmConfig(num_labels=len(label_list), token_classifier_type=classifier_type)
        model = BertCausalmForTokenClassification(config)
    else:
        config = BertCausalmConfig(num_labels=len(label_list), sequence_classifier_type=classifier_type)
        model = BertCausalmForSequenceClassification(config)

    model.bert = bert_model

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

    labels_to_ignore = {'cc_labels', 'tc_labels', 'task_labels'}
    labels_to_ignore.remove(f'{classifier_type}_labels')
    dataset = dataset.remove_columns(list(labels_to_ignore))

    if classifier_type == 'tc' or classifier_type == 'cc':
        data_collator = DataCollatorForCausalmTokenClassification(tokenizer, label_name=label_name, labels_to_ignore=labels_to_ignore)
        metric = load_metric('seqeval')

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
    elif classifier_type == 'task':
        data_collator = None
        metric = load_metric('accuracy')

        def compute_metrics(p):
            preds, labels = p
            preds = np.argmax(preds, axis=1)
            return metric.compute(predictions=preds, references=labels)
    else:
        raise NotImplementedError()

    args._n_gpu = 1

    trainer = CausalmTrainer(
        model,
        args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['train'] if overfit else dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if skip_training:
        trainer.evaluate()
    else:
        trainer.train()

    eval_results = trainer.evaluate()

    return trainer.model, eval_results


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

    datasets = datasets.remove_columns(['ps', 'review_text'])

    if cf:
        datasets = datasets.remove_columns(['tokens'])
        datasets = datasets.rename_column('ps_cf', 'tokens')
    datasets = datasets.rename_column('ps_tags', 'tc_labels')
    datasets = datasets.rename_column('ner_tags', 'cc_labels')
    datasets = datasets.rename_column('sentiment', 'task_labels')

    labels_names = ['cc_labels', 'tc_labels']

    datasets = datasets.map(tokenize_and_align_labels, batched=True,
                            fn_kwargs={'tokenizer': tokenizer, 'label_names': labels_names, 'text_key': 'tokens'})

    return datasets


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
    task_label_list = dataset_f['train'].features['task_labels'].names

    # 1) Pretraining
    bert_o = BertModel.from_pretrained(BERT_MODEL_CHECKPOINT)

    # 2) Additional Pretraining
    bert_cf = additional_pretraining_pipeline(tokenizer, dataset_f['train'], epochs=2)

    # 3) Downstream Task Training
    bert_o_task_classifier, bert_o_task_classifier_metrics = downstream_task_pipeline(
        dataset_f, bert_o, 'task', label_list=task_label_list, epochs=4)
    bert_cf_task_classifier, bert_cf_task_classifier_metrics = downstream_task_pipeline(
        dataset_f, bert_cf, 'task', label_list=task_label_list, epochs=4)

    # 4a) Probing for Treated Concept
    _, bert_o_tc_classifier_metrics = downstream_task_pipeline(dataset_f, bert_o, 'tc', label_list=tc_label_list, epochs=4)
    _, bert_cf_tc_classifier_metrics = downstream_task_pipeline(dataset_f, bert_cf, 'tc', label_list=tc_label_list, epochs=4)

    # 4b) Probing for Control Concept
    _, bert_o_cc_classifier_metrics = downstream_task_pipeline(dataset_f, bert_o, 'cc', label_list=cc_label_list, epochs=4)
    _, bert_cf_cc_classifier_metrics = downstream_task_pipeline(dataset_f, bert_cf, 'cc', label_list=cc_label_list, epochs=4)

    # 5) ATEs Estimation
    metrics_cls = CausalmMetrics(text_key='tokens', tokenizer_checkpoint=BERT_MODEL_CHECKPOINT)
    treate = metrics_cls.treate(model_o=bert_o_task_classifier, model_cf=bert_cf_task_classifier, dataset=dataset_f['test'])
    ate = metrics_cls.ate(model=bert_o_task_classifier, dataset_f=dataset_f['test'], dataset_cf=dataset_cf['test'])
    conexp = metrics_cls.conexp(model=bert_o_task_classifier, dataset=dataset_f['test'])

    # save results
    results_df = pandas.DataFrame.from_records({
        'Task': ['Sentiment'],
        'Adversarial Task': ['Product Specific'],
        'Control Task': ['NER'],
        'Confounder': ['Domain'],

        'BERT-O Treated Performance': [bert_o_tc_classifier_metrics['eval_accuracy']],
        'BERT-CF Treated Performance': [bert_cf_tc_classifier_metrics['eval_accuracy']],
        'INLP Treated Performance': [None],

        'BERT-O Control Performance': [bert_o_cc_classifier_metrics['eval_accuracy']],
        'BERT-CF Control Performance': [bert_cf_cc_classifier_metrics['eval_accuracy']],
        'INLP Control Performance': [None],

        'BERT-O Task Performance': [bert_o_task_classifier_metrics['eval_accuracy']],
        'BERT-CF Task Performance': [bert_cf_task_classifier_metrics['eval_accuracy']],
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
