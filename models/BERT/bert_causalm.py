from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
import torch.nn as nn

from models.BERT.configuration_causalm import BertCausalmConfig, CausalmHeadConfig
from utils import SEQUENCE_CLASSIFICATION, TOKEN_CLASSIFICATION


class MaskedLMCausalmHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlm_head = BertLMPredictionHead(config)
        self.tc_heads = self.__init_causalm_heads(config, 'tc')
        self.cc_heads = self.__init_causalm_heads(config, 'cc')

    def forward(self, sequence_output):
        mlm_head_scores = self.mlm_head(sequence_output)
        tc_heads_scores = [tc_head(sequence_output) for tc_head in self.tc_heads]
        cc_heads_scores = [cc_head(sequence_output) for cc_head in self.cc_heads]
        return mlm_head_scores, tc_heads_scores, cc_heads_scores

    @staticmethod
    def __init_causalm_heads(config, mode):
        heads = []

        if mode == 'tc':
            heads_cfg = config.tc_heads_cfg
        elif mode == 'cc':
            heads_cfg = config.cc_heads_cfg
        else:
            raise RuntimeError(f'Illegal mode: "{mode}". Can be either "tc" or "cc".')

        for head_cfg in heads_cfg:
            if head_cfg.head_type == SEQUENCE_CLASSIFICATION:
                heads.append(nn.Sequential(
                    nn.Dropout(head_cfg.hidden_dropout_prob),
                    nn.Linear(config.hidden_size, head_cfg.num_labels)
                ))
            elif head_cfg.head_type == TOKEN_CLASSIFICATION:
                raise NotImplementedError()
        return heads


class CaribbeanBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.additional_pretraining_heads = MaskedLMCausalmHeads(config)

        self.init_weights()

    def forward(self, sequence_output):
        print('forward!')


def test_caribbean_bert():
    config = BertCausalmConfig(
        tc_heads_cfg=[CausalmHeadConfig(head_type=SEQUENCE_CLASSIFICATION,
                                        head_params={'hidden_dropout_prob': 0.0, 'num_labels': 2})],
        cc_heads_cfg=[CausalmHeadConfig(head_type=SEQUENCE_CLASSIFICATION,
                                        head_params={'hidden_dropout_prob': 0.0, 'num_labels': 2})]
    )
    caribbean_bert = CaribbeanBert(config)

    caribbean_bert(5)
