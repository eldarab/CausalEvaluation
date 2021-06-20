import time

import numpy as np
from datasets import load_metric

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling, AdamW, BertTokenizer, BertModel, TrainingArguments, Trainer

from experiments.sentiment_acceptability_domain.dataset import SADDataset
from modeling.BERT.bert_causalm import BertForCausalmAdditionalPreTraining, BertCausalmForSequenceClassification
from modeling.BERT.configuration_causalm import BertCausalmConfig, CausalmHeadConfig
from utils import DATA_DIR, BERT_MODEL_CHECKPOINT, SEQUENCE_CLASSIFICATION, DEVICE, PROJECT_DIR


def additional_pretraining_pipeline():
    time_str = time.strftime('%Y_%m_%d__%H_%M_%S')

    # data
    train_dataset = SADDataset(data_dir=f'{DATA_DIR}/acceptability_sample.csv', fold='train')
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_CHECKPOINT)
    lm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=lm_data_collator)

    # model
    config = BertCausalmConfig(
        tc_heads_cfg=[CausalmHeadConfig(head_name='acceptability', head_type=SEQUENCE_CLASSIFICATION, head_params={'num_labels': 2})],
        cc_heads_cfg=[CausalmHeadConfig(head_name='is_books', head_type=SEQUENCE_CLASSIFICATION, head_params={'num_labels': 2})],
        tc_lambda=0.2,
    )
    model = BertForCausalmAdditionalPreTraining(config)
    model.to(DEVICE)
    model.train()

    optim = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(10):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'epoch {epoch:3d}'):
            optim.zero_grad()

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            lm_labels = batch['labels'].to(DEVICE)
            tc_labels = batch['tc_label'].to(DEVICE)
            cc_labels = batch['cc_label'].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, lm_labels=lm_labels, tc_labels=tc_labels, cc_labels=cc_labels)

            loss = outputs[0]
            total_loss += loss / len(batch)

            loss.backward()
            optim.step()
        print(f"loss: {total_loss:.3f}")

    save_dir = f'{PROJECT_DIR}/saved_models/SAD__{time_str}'
    model.bert.save_pretrained(save_directory=save_dir)
    print(f'Saved model.bert in {save_dir}')


def downstream_task_training_pipeline(
        train_dataset,
        test_dataset,
        model_checkpoint=BERT_MODEL_CHECKPOINT,
        num_labels=2,
        metric_name='accuracy',
        lr=2e-5,
        epochs=10,
        overfit=False,
        bert_cf=None,
        skip_training=False,
):
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint)

    config = BertCausalmConfig(num_labels=num_labels, sequence_classifier_type='task')
    model = BertCausalmForSequenceClassification(config)

    if bert_cf:
        model.bert = bert_cf
    else:
        model.bert = BertModel.from_pretrained(model_checkpoint)

    metric = load_metric(metric_name)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labels)

    # noinspection PyTypeChecker
    args = TrainingArguments(
        'sanity_check',
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
        eval_dict = trainer.evaluate()
        print(eval_dict)
    else:
        print("\n============ before training ============")
        print(trainer.evaluate())
        trainer.train()
        print("\n============ after training ============")
        eval_dict = trainer.evaluate()
        print(eval_dict)

    return eval_dict['eval_accuracy']


if __name__ == '__main__':
    # compute CONEXP
    bert_o_acceptable_accuracy = downstream_task_training_pipeline(
        train_dataset=SADDataset(data_dir=DATA_DIR, fold='train'),
        test_dataset=SADDataset(data_dir=DATA_DIR, fold='test', textual_counterfactual=True, conexp_fold='acceptable'),
        bert_cf=None,
        skip_training=True
    )

    bert_o_unacceptable_accuracy = downstream_task_training_pipeline(
        train_dataset=SADDataset(data_dir=DATA_DIR, fold='train'),
        test_dataset=SADDataset(data_dir=DATA_DIR, fold='test', textual_counterfactual=True, conexp_fold='unacceptable'),
        bert_cf=None,
        skip_training=True
    )

    # compute TReATE
    bert_cf_factual_accuracy = downstream_task_training_pipeline(
        train_dataset=SADDataset(data_dir=DATA_DIR, fold='train'),
        test_dataset=SADDataset(data_dir=DATA_DIR, fold='test'),
        overfit=False,
        bert_cf=BertModel.from_pretrained(f'{PROJECT_DIR}/saved_models/SAD__2021_06_15__09_31_17')
    )

    # compute ATE_gt
    bert_o_factual_accuracy = downstream_task_training_pipeline(
        train_dataset=SADDataset(data_dir=DATA_DIR, fold='train'),
        test_dataset=SADDataset(data_dir=DATA_DIR, fold='test'),
        bert_cf=None
    )

    bert_o_counterfactual_accuracy = downstream_task_training_pipeline(
        train_dataset=SADDataset(data_dir=DATA_DIR, fold='train'),
        test_dataset=SADDataset(data_dir=DATA_DIR, fold='test', textual_counterfactual=True),
        bert_cf=None
    )

    treate = bert_o_factual_accuracy - bert_cf_factual_accuracy
    ate_gt = bert_o_factual_accuracy - bert_o_counterfactual_accuracy
    conexp = bert_o_acceptable_accuracy - bert_o_unacceptable_accuracy
    print('\n\n\n\n')
    print(f'TReATE: {treate}')
    print(f'ATE_gt: {ate_gt}')
    print(f'CONEXP: {conexp}')
