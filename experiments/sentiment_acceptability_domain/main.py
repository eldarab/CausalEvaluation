from datasets import load_dataset, ClassLabel, Features, Value


def main():
    features = Features({
        'text': Value(dtype='string', id='text'),
        'acceptability_sophiemarshall2': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], id='acceptability'),
        'is_books': ClassLabel(num_classes=2, names=['not_books', 'books'], id='is_books'),
        'sentiment': ClassLabel(num_classes=2, names=['negative', 'positive'], id='sentiment'),
    })
    dataset = load_dataset('csv', data_files=['../../data/acceptability_sample.csv'], index_col=0,
                           features=features)
    for sample in dataset['train']:
        for k, v in sample.items():
            print(f'{k:<35}{v}')
        print()



if __name__ == '__main__':
    main()