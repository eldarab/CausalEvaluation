from typing import Dict, Union, Any, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.cuda.amp import autocast
from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.trainer import Trainer, TrainingArguments
from transformers.trainer_pt_utils import nested_detach
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward
from dataclasses import field, dataclass


class CausalmTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all modeling return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        elif self.label_smoother is not None and self.label_names[0] in inputs:
            labels = inputs.pop(self.label_names[0])
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
                Perform an evaluation step on :obj:`model` using obj:`inputs`.

                Subclass and override to inject custom behavior.

                Args:
                    model (:obj:`nn.Module`):
                        The model to evaluate.
                    inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                        The inputs and targets of the model.

                        The dictionary will be unpacked before being fed to the model. Most modeling expect the targets under the
                        argument :obj:`labels`. Check your model's documentation for all accepted arguments.
                    prediction_loss_only (:obj:`bool`):
                        Whether or not to return the loss only.
                    ignore_keys (:obj:`Lst[str]`, `optional`):
                        A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                        gathering predictions.

                Return:
                    Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                    logits and labels (each being optional).
                """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return loss, None, None

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return loss, logits, labels

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.use_amp else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def train_causalm_additional_pretraining(self, tc, cc):
        """
        TODO: Implement train_causalm_additional_pretraining
        Ultimately, this function will replace 'additional_pretraining_pipeline()' from product_ner_domain.py.
        """
        raise NotImplementedError()


@dataclass
class CausalmTrainingArguments(TrainingArguments):
    causalm_additional_pretraining: bool = field(
        default=False,
        metadata={"help": "Whether or not this CausaLM trainer performs additional pretraining."},
    )
    num_cc: int = field(
        default=0,
        metadata={'help': 'Number of Control Concepts this trainer trains.'}
    )
    num_tc: int = field(
        default=0,
        metadata={'help': 'Number of Treatment Concepts this trainer trains.'}
    )
