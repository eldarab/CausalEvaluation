from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import BertTokenizerFast
from transformers.data.data_collator import _collate_batch

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from .constants import BERT_MODEL_CHECKPOINT


@dataclass
class DataCollatorForCausalmAdditionalPretraining:

    tokenizer: PreTrainedTokenizerBase
    collate_tc: bool = False
    collate_cc: bool = False
    mlm: bool = True
    mlm_probability: float = 0.15
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __post_init__(  # DataCollatorForLanguageModeling
            self
    ):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(  # DataCollatorForTokenClassification
            self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        if self.collate_tc:
            tc_label_name = 'tc_label' if 'tc_label' in examples[0].keys() else 'tc_labels'
            tc_token_labels = [feature[tc_label_name] for feature in examples] if tc_label_name in examples[0].keys() else None
        else:
            tc_label_name, tc_token_labels = None, None

        if self.collate_cc:
            cc_label_name = 'cc_label' if 'cc_label' in examples[0].keys() else 'cc_labels'
            cc_token_labels = [feature[cc_label_name] for feature in examples] if cc_label_name in examples[0].keys() else None
        else:
            cc_label_name, cc_token_labels = None, None

        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(
                examples,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                # Conversion to tensors will fail if we have labels as they are not of the same length yet.
                return_tensors="pt",
            )
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}
            raise NotImplementedError()

        # MLM STUFF ----------------
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["mlm_labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["mlm_labels"] = labels
        # --------------------------

        if tc_token_labels is None and cc_token_labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            if tc_label_name:
                batch[tc_label_name] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in tc_token_labels]
            if cc_label_name:
                batch[cc_label_name] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in cc_token_labels]
        else:  # padding_size == "left"
            if tc_label_name:
                batch[tc_label_name] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in tc_token_labels]
            if cc_label_name:
                batch[cc_label_name] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in cc_token_labels]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}

        return batch

    def mask_tokens(  # DataCollatorForLanguageModeling
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_CHECKPOINT)

    data_collator = DataCollatorForCausalmAdditionalPretraining(tokenizer=tokenizer)
