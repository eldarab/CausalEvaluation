import torch
from datasets import load_dataset, ClassLabel, Features, Value
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from utils import DATA_DIR, RANDOM_SEED


class CaribbeanDataset(Dataset):
    def __init__(self, data_path, fold='train'):
        features = Features({
            'text': Value(dtype='string', id='text'),
            'acceptability_sophiemarshall2': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], id='acceptability'),
            'is_books': ClassLabel(num_classes=2, names=['not_books', 'books'], id='is_books'),
            'sentiment': ClassLabel(num_classes=2, names=['negative', 'positive'], id='sentiment'),
        })
        dataset = load_dataset('csv', data_files=[data_path], index_col=0, features=features)
        dataset = dataset['train'].train_test_split(test_size=0.2, seed=RANDOM_SEED)
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        self.encodings = tokenizer(dataset[fold]['text'], truncation=True, padding=True)
        self.task_labels = dataset[fold]['sentiment']
        self.tc_labels = dataset[fold]['acceptability_sophiemarshall2']
        self.cc_labels = dataset[fold]['is_books']

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['task_label'] = torch.tensor(self.task_labels[idx])
        item['tc_label'] = torch.tensor(self.tc_labels[idx])
        item['cc_label'] = torch.tensor(self.cc_labels[idx])
        return item

    def __len__(self):
        return len(self.task_labels)


def test_caribbean_dataset():
    train_dataset = CaribbeanDataset(data_path=f'{DATA_DIR}/acceptability_sample.csv', fold='train')
    test_dataset = CaribbeanDataset(data_path=f'{DATA_DIR}/acceptability_sample.csv', fold='test')

    print(train_dataset[0])
    print(test_dataset[0])


if __name__ == '__main__':
    test_caribbean_dataset()
