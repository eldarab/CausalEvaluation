import torch
from datasets import load_dataset, ClassLabel, Features, Value
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, Trainer
from transformers.trainer import Trainer
from utils import DATA_DIR, RANDOM_SEED


class CaribbeanDataset(Dataset):
    def __init__(self, encodings, task_labels, cc_labels, tc_labels):
        self.encodings = encodings
        self.task_labels = task_labels
        self.tc_labels = tc_labels
        self.cc_labels = cc_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['task_labels'] = torch.tensor(self.task_labels[idx])
        item['tc_labels'] = torch.tensor(self.tc_labels[idx])
        item['cc_labels'] = torch.tensor(self.cc_labels[idx])
        return item

    def __len__(self):
        return len(self.task_labels)


def test_caribbean_dataset():
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

    print(train_dataset[0])
    print(test_dataset[0])


if __name__ == '__main__':
    test_caribbean_dataset()
