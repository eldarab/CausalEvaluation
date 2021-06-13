import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizerFast, AdamW, DataCollatorForLanguageModeling

from experiments.sentiment_acceptability_domain.dataset import CaribbeanDataset
from models.BERT.bert_causalm import BertForCausalmAdditionalPreTraining
from models.BERT.configuration_causalm import BertCausalmConfig, CausalmHeadConfig
from utils import DATA_DIR, PROJECT_DIR
from utils import SEQUENCE_CLASSIFICATION


def main():
    # technicalities
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    time_str = time.strftime('%Y_%m_%d__%H_%M_%S')

    # data
    train_dataset = CaribbeanDataset(data_path=f'{DATA_DIR}/acceptability_sample.csv', fold='train')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    lm_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lm_data_collator)

    # model
    config = BertCausalmConfig(
        tc_heads_cfg=[CausalmHeadConfig(head_name='acceptability', head_type=SEQUENCE_CLASSIFICATION, head_params={'num_labels': 2})],
        cc_heads_cfg=[CausalmHeadConfig(head_name='is_books', head_type=SEQUENCE_CLASSIFICATION, head_params={'num_labels': 2})],
        tc_lambda=-0.2,
    )
    model = BertForCausalmAdditionalPreTraining(config)
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(10):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'epoch: {epoch:3d}'):
            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            lm_labels = batch['labels'].to(device)
            tc_labels = batch['tc_label'].to(device)
            cc_labels = batch['cc_label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, lm_labels=lm_labels, tc_labels=tc_labels, cc_labels=cc_labels)

            loss = outputs[0]
            total_loss += loss / len(batch)

            loss.backward()
            optim.step()
        print(f"loss: {total_loss:.3f}")

    save_dir = f'{PROJECT_DIR}/saved_models/sentiment_acceptability_domain__{time_str}'
    model.bert.save_pretrained(save_directory=save_dir)
    print(f'Saved model.bert in {save_dir}')


if __name__ == '__main__':
    main()
