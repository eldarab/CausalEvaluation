from transformers.models.bert.configuration_bert import BertConfig
import torch.nn as nn

from typing import List
from utils import HEAD_TYPES, SEQUENCE_CLASSIFICATION, TOKEN_CLASSIFICATION


class CausalmHeadConfig:
    def __init__(self, head_type, head_params: dict):
        if head_type not in HEAD_TYPES:
            raise RuntimeError(f'Illegal head type: "{head_type}"')

        self.head_type = head_type

        if head_type == SEQUENCE_CLASSIFICATION:
            self.hidden_dropout_prob = head_params.pop('hidden_dropout_prob')
            self.num_labels = head_params.pop('num_labels')

        if head_type == TOKEN_CLASSIFICATION:
            raise NotImplementedError()


class BertCausalmConfig(BertConfig):
    """
    Adds a tc_heads and cc_heads parameters to config.
    """
    model_type = "bert_causalm"

    def __init__(
            self,
            tc_heads_cfg: List[CausalmHeadConfig],
            cc_heads_cfg: List[CausalmHeadConfig],
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,
            position_embedding_type="absolute",
            use_cache=True,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache

        # causalm configs
        self.tc_heads_cfg = tc_heads_cfg
        self.cc_heads_cfg = cc_heads_cfg
