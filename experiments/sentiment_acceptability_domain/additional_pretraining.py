import time

import torch
from datasets import load_dataset, ClassLabel, Features, Value
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast, AdamW, BertForPreTraining, BertConfig, DataCollatorForLanguageModeling

from experiments.sentiment_acceptability_domain.dataset import CaribbeanDataset
from models.BERT.bert_causalm import BertForCausalmAdditionalPreTraining
from models.BERT.configuration_causalm import BertCausalmConfig, CausalmHeadConfig
from utils import DATA_DIR, RANDOM_SEED, SENTIMENT_ACCEPTABILITY_DOMAIN_DIR
from utils import SEQUENCE_CLASSIFICATION


def main():
    # torch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    time_str = time.strftime('%Y_%m_%d__%H_%M_%S')

    # data
    features = Features({
        'text': Value(dtype='string', id='text'),
        'acceptability_sophiemarshall2': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], id='acceptability'),
        'is_books': ClassLabel(num_classes=2, names=['not_books', 'books'], id='is_books'),
        'sentiment': ClassLabel(num_classes=2, names=['negative', 'positive'], id='sentiment'),
    })
    dataset = load_dataset('csv', data_files=[f'{DATA_DIR}/acceptability_sample.csv'], index_col=0, features=features)
    dataset = dataset['train'].train_test_split(test_size=0.2, seed=RANDOM_SEED)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(dataset['train']['text'], truncation=True, padding=True)
    test_encodings = tokenizer(dataset['test']['text'], truncation=True, padding=True)

    train_dataset = CaribbeanDataset(
        encodings=train_encodings,
        task_labels=dataset['train']['sentiment'],
        tc_labels=dataset['train']['acceptability_sophiemarshall2'],
        cc_labels=dataset['train']['is_books'],
    )
    test_dataset = CaribbeanDataset(
        encodings=test_encodings,
        task_labels=dataset['test']['sentiment'],
        tc_labels=dataset['test']['acceptability_sophiemarshall2'],
        cc_labels=dataset['test']['is_books'],
    )

    # model
    config = BertCausalmConfig(
        tc_heads_cfg=[CausalmHeadConfig(head_name='acceptability', head_type=SEQUENCE_CLASSIFICATION, head_params={'num_labels': 2})],
        cc_heads_cfg=[CausalmHeadConfig(head_name='is_books', head_type=SEQUENCE_CLASSIFICATION, head_params={'num_labels': 2})],
    )
    model = BertForCausalmAdditionalPreTraining(config)
    model.to(device)
    model.train()

    lm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lm_data_collator)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(5):
        total_loss = 0
        for batch in tqdm(train_loader, desc='batch'):
            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lm_labels = batch['labels'].to(device)
            tc_labels = batch['tc_labels'].to(device)
            cc_labels = batch['cc_labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, lm_labels=lm_labels, tc_labels=tc_labels, cc_labels=cc_labels)

            loss = outputs[0]
            total_loss += loss / len(batch)

            loss.backward()
            optim.step()
        print(f"epoch: {epoch:3d} loss: {total_loss:.3f}")

    model.bert.save_pretrained(save_directory=f'{SENTIMENT_ACCEPTABILITY_DOMAIN_DIR}/saved_models/bert__{time_str}')


if __name__ == '__main__':
    main()
