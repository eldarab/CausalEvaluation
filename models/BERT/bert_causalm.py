from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.file_utils import ModelOutput

from models.BERT.configuration_causalm import BertCausalmConfig, CausalmHeadConfig
from utils import SEQUENCE_CLASSIFICATION, TOKEN_CLASSIFICATION


@dataclass
class BertForCausalmAdditionalPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    tc_logits: torch.FloatTensor = None
    cc_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertCausalmAdditionalPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlm_head = BertLMPredictionHead(config)
        self.tc_heads = self.__init_causalm_heads(config, 'tc')
        self.cc_heads = self.__init_causalm_heads(config, 'cc')

    def forward(self, sequence_output, pooled_output):
        mlm_head_scores = self.mlm_head(sequence_output)
        tc_heads_scores = [tc_head(pooled_output) for tc_head in self.tc_heads]
        cc_heads_scores = [cc_head(pooled_output) for cc_head in self.cc_heads]
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


class BertForCausalmAdditionalPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        self.additional_pretraining_heads = BertCausalmAdditionalPreTrainingHeads(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,  # TODO TC/CC label
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        mlm_head_scores, tc_heads_scores, cc_heads_scores = self.additional_pretraining_heads(sequence_output, pooled_output)

        # TODO add token classification version
        for head in self.config.tc_heads_cfg + self.config.cc_heads_cfg:
            if head.head_type != SEQUENCE_CLASSIFICATION:
                raise NotImplementedError()

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()

            masked_lm_loss = loss_fct(mlm_head_scores.view(-1, self.config.vocab_size), labels.view(-1))

            tc_loss = torch.tensor(0.)
            for tc_head_score, tc_head_cfg in zip(tc_heads_scores, self.config.tc_heads_cfg):
                tc_loss += loss_fct(tc_head_score.view(-1, tc_head_cfg.num_labels), next_sentence_label.view(-1))

            cc_loss = torch.tensor(0.)
            for cc_head_score, cc_head_cfg in zip(cc_heads_scores, self.config.cc_heads_cfg):
                cc_loss += loss_fct(cc_head_score.view(-1, cc_head_cfg.num_labels), next_sentence_label.view(-1))

            total_loss = masked_lm_loss + tc_loss - cc_loss  # note the loss reversal!

        if not return_dict:
            output = (mlm_head_scores, tc_heads_scores, cc_heads_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForCausalmAdditionalPreTrainingOutput(
            loss=total_loss,
            mlm_logits=mlm_head_scores,
            tc_logits=tc_heads_scores,
            cc_logits=cc_heads_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def test_caribbean_bert():
    config = BertCausalmConfig(
        tc_heads_cfg=[CausalmHeadConfig(head_type=SEQUENCE_CLASSIFICATION,
                                        head_params={'hidden_dropout_prob': 0.0, 'num_labels': 2})],
        cc_heads_cfg=[CausalmHeadConfig(head_type=SEQUENCE_CLASSIFICATION,
                                        head_params={'hidden_dropout_prob': 0.0, 'num_labels': 2})]
    )
    caribbean_bert = BertForCausalmAdditionalPreTraining(config)

    caribbean_bert(5)
