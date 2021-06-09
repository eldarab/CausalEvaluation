# Supervised task training: Where task specific layers are trained on labeled data for a downstream
# task of interest.
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertModel

from experiments.sentiment_acceptability_domain.dataset import CaribbeanDataset
from models.BERT.configuration_causalm import BertCausalmConfig
from utils import DATA_DIR, CAUSAL_EVAL_DIR


def main():
    train_dataset = CaribbeanDataset(data_path=f'{DATA_DIR}/acceptability_sample.csv', fold='train')
    # test_dataset = CaribbeanDataset(data_path=f'{DATA_DIR}/acceptability_sample.csv', fold='test')

    save_path = f'{CAUSAL_EVAL_DIR}/saved_models/sentiment_acceptability_domain__2021_06_06__18_52_12'
    config = BertCausalmConfig.from_pretrained(save_path)
    model = BertForSequenceClassification(config)
    model.bert = BertModel.from_pretrained(save_path)


if __name__ == '__main__':
    main()
