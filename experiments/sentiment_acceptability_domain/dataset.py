import torch
import pandas as pd
from torch.utils.data import Dataset as PyTorchDataset
from transformers import BertTokenizerFast
from pathlib import Path
from datasets import Dataset as HuggingFaceDataset

from utils import DATA_DIR, BERT_MODEL_CHECKPOINT


class SADDataset(PyTorchDataset):
    def __init__(
            self,
            data_dir: str,
            fold: str = 'train',
            textual_counterfactual=False,
            conexp_fold=None
    ):
        data_dir = Path(data_dir)

        if fold not in {'train', 'val', 'test'}:
            raise RuntimeError(f'Illegal fold "{fold}"')

        if textual_counterfactual and fold != 'test':
            raise RuntimeError(f'There exists no textual counterfactuals for fold "{fold}"')

        df = pd.read_csv(data_dir / f'SAD_{fold}.csv', index_col=0)

        if fold == 'test' and conexp_fold == 'acceptable':
            df = df[df['acceptability_sophiemarshall2'] == 1]
        if fold == 'test' and conexp_fold == 'unacceptable':
            df = df[df['acceptability_sophiemarshall2'] == 0]

        tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_CHECKPOINT)

        self.conexp_fold = conexp_fold
        self.textual_counterfactual = textual_counterfactual
        self.fold = fold
        if fold == 'test' and self.textual_counterfactual:
            self.text_acceptability_cf = df['acceptability_cf_amitavasil']
            self.encodings_acceptability_textual_cf = tokenizer(list(df['acceptability_cf_amitavasil']), truncation=True, padding=True)
        else:
            self.text = df['text']
            self.encodings = tokenizer(list(df['text']), truncation=True, padding=True)
        self.task_labels = df['sentiment']
        self.tc_labels = df['acceptability_sophiemarshall2']
        self.cc_labels = df['is_books']

    def __getitem__(self, idx):
        if self.fold == 'test' and self.textual_counterfactual:
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings_acceptability_textual_cf.items()}
        else:
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['task_labels'] = torch.tensor(self.task_labels[idx])
        item['tc_labels'] = torch.tensor(self.tc_labels[idx])
        item['cc_labels'] = torch.tensor(self.cc_labels[idx])
        return item

    def __len__(self):
        return len(self.task_labels)


def gen_caribbean_dataset():
    train_dataset = SADDataset(data_dir=DATA_DIR, fold='train')
    test_dataset = SADDataset(data_dir=DATA_DIR, fold='test')

    print(train_dataset[0])
    print(test_dataset[0])


if __name__ == '__main__':
    gen_caribbean_dataset()
