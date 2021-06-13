# Check that the model forgot the TC and remembered MLM and CC
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers.models.auto.tokenization_auto import BertTokenizerFast
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertModel

from experiments.sentiment_acceptability_domain.dataset import CaribbeanDataset
from models.BERT.configuration_causalm import BertCausalmConfig
from utils import DATA_DIR, PROJECT_DIR
from utils.metrics import calc_accuracy_from_logits


def main(args):
    # torch
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    # data
    train_dataset = CaribbeanDataset(data_path=f'{DATA_DIR}/acceptability_sample.csv', fold='train')
    test_dataset = CaribbeanDataset(data_path=f'{DATA_DIR}/acceptability_sample.csv', fold='test')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, drop_last=True)

    # config
    save_path = f'{PROJECT_DIR}/saved_models/sentiment_acceptability_domain__2021_06_06__18_52_12'
    config = BertCausalmConfig.from_pretrained(save_path)
    config.num_labels = 2
    config.id2label = {0: 'unacceptable', 1: 'acceptable'}
    config.label2id = {'unacceptable': 0, 'acceptable': 1}
    config.problem_type = 'single_label_classification'

    # model
    model = BertForSequenceClassification(config)
    model.to(device)
    model.train()

    print('======== CONTROL MODEL ========')
    train(model, train_loader, device, test_loader)

    model.bert = BertModel.from_pretrained(save_path)
    model.to(device)
    model.train()

    print('======== TREATED MODEL ========')
    train(model, train_loader, device, test_loader)

    base_test_acc = sum([item['tc_label'] for item in test_dataset]) / len(test_dataset)
    print(f'Baseline test accuracy: {base_test_acc:.3f} for {len(test_dataset)} eval samples.')


def train(model, train_loader, device, test_loader):
    optim = AdamW(model.parameters(), lr=5e-4)

    for epoch in range(20):
        train_loss = 0
        train_acc = 0
        for batch in tqdm(train_loader, desc=f'epoch {epoch}'):
            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            tc_labels = batch['tc_label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=tc_labels)

            loss = outputs[0]
            with torch.no_grad():
                train_acc += calc_accuracy_from_logits(outputs=outputs[1], true_labels=tc_labels, model=model)[0] / len(train_loader)
            train_loss += loss / len(batch)

            loss.backward()
            optim.step()

        with torch.no_grad():
            eval_acc = 0
            eval_loss = 0
            for batch in tqdm(test_loader, desc='eval'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                tc_labels = batch['tc_label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=tc_labels)

                eval_loss += outputs[0] / len(batch)
                eval_acc += calc_accuracy_from_logits(outputs=outputs[1], true_labels=tc_labels, model=model)[0] / len(test_loader)

        print(f"train loss: {train_loss:.3f} train accuracy: {train_acc:.3f}\n"
              f"eval loss:  {eval_loss:.3f}  eval accuracy:  {eval_acc:.3f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True, help='The name of the model to load.')
    args = parser.parse_args()

    main(args)
