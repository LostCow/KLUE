import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from typing import Optional, Tuple
from dataclasses import dataclass

from transformers.modeling_outputs import QuestionAnsweringModelOutput, SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaClassificationHead,
)
from transformers.file_utils import ModelOutput


@dataclass
class QuestionAnsweringAVPoolModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    has_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # sketch_reader
    # is_answerable, unanswerable

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaForQuestionAnsweringAVPool(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # intensive_reader

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.has_ans = nn.Sequential(
            nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 4)
        )
        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cls_idx=None,
    ):

        outputs = self.roberta(
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

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        first_word = sequence_output[:, 0, :]
        # first_word = outputs[1]

        # batch * hidden_dim * max_seq_length

        # has_inp = torch.cat([p_avg, first_word, q_summ, p_avg_p], -1)
        has_log = self.has_ans(first_word)  # batch * hidden_dim * 1

        outputs = (
            start_logits,
            end_logits,
            has_log,
        ) + outputs[2:]

        total_loss = None
        # print(has_log)
        # print(cls_idx)
        if start_positions is not None and end_positions is not None and cls_idx is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            cls_loss_fct = CrossEntropyLoss()
            choice_loss = cls_loss_fct(has_log, cls_idx)

            total_loss = (start_loss + end_loss) / 2 + choice_loss
            outputs = (total_loss,) + outputs

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringAVPoolModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            has_logits=has_log,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class RobertaCNNForQuestionAnsweringAVPool(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # intensive_reader

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.has_ans = nn.Sequential(
            nn.Dropout(p=config.hidden_dropout_prob), nn.Linear(config.hidden_size, 4)
        )
        self.init_weights()

        self.hidden_dim = config.hidden_size

        # self.qa_outputs = nn.Linear(in_features=384 * 3, out_features=config.num_labels)
        self.qa_outputs = nn.Linear(in_features=self.hidden_dim * 3, out_features=config.num_labels)

        self.conv1d_k1 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=1024, kernel_size=1, padding=0)
        self.conv1d_k3 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=1024, kernel_size=3, padding=1)
        self.conv1d_k5 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=1024, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cls_idx=None,
    ):

        outputs = self.roberta(
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

        sequence_output = outputs[0]
        cnn_input = sequence_output.permute(0, 2, 1)
        # print(f"{sequence_output.shape=}")

        cnn_k1_output = self.relu(self.conv1d_k1(cnn_input))
        cnn_k3_output = self.relu(self.conv1d_k3(cnn_input))
        cnn_k5_output = self.relu(self.conv1d_k5(cnn_input))
        concat_cnn_output = torch.cat((cnn_k1_output, cnn_k3_output, cnn_k5_output), 1)

        logits = self.qa_outputs(concat_cnn_output.permute(0, 2, 1))

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        first_word = sequence_output[:, 0, :]

        has_log = self.has_ans(first_word)  # batch * hidden_dim * 1

        outputs = (
            start_logits,
            end_logits,
            has_log,
        ) + outputs[2:]

        total_loss = None
        # print(has_log)
        # print(cls_idx)
        if start_positions is not None and end_positions is not None and cls_idx is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)

            cls_loss_fct = CrossEntropyLoss()
            choice_loss = cls_loss_fct(has_log, cls_idx)

            total_loss = (start_loss + end_loss) / 2 + choice_loss
            outputs = (total_loss,) + outputs

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringAVPoolModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            has_logits=has_log,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
