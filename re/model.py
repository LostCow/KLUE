import torch
from torch import nn
from transformers import (
    RobertaModel,
    RobertaPreTrainedModel,
    BertModel,
    BertPreTrainedModel,
)
import torch.nn.functional as F
import numpy as np
from losses import CB_loss


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class Rbert(RobertaPreTrainedModel):
    """
    orgin code: https://github.com/monologg/R-BERT
    edit by 문하겸_T2076
    """

    def __init__(self, config):
        super(Rbert, self).__init__(config)
        self.roberta = RobertaModel(config=config)  # Load pretrained model

        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.dense = FCLayer(config.hidden_size * 3, config.hidden_size, 0.1)

        self.label_classifier = FCLayer(
            config.hidden_size,
            config.num_labels,
            0.1,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        e1_mask=None,
        e2_mask=None,
    ):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        concat_h = self.dense(concat_h)
        logits = self.label_classifier(concat_h)
        # print(outputs)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        # print(outputs)

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs
        return outputs


class SkeletonAwareBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(SkeletonAwareBERT, self).__init__(config)
        self.bert = BertModel(config=config)  # Load pretrained model

        self.num_labels = config.num_labels

        # self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.entity_fc_layer1 = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.entity_fc_layer2 = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.si_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.ctx_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)

        self.dense = FCLayer(config.hidden_size * 4, config.hidden_size, 0.1)

        self.label_classifier = FCLayer(
            config.hidden_size,
            config.num_labels,
            0.1,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        e1_mask=None,
        e2_mask=None,
        si_mask=None,
        ctx_mask=None,
    ):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        # print(sequence_output)
        # print(pooled_output)
        # print(sequence_output)
        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        si_h = self.entity_average(sequence_output, si_mask)
        ctx_h = self.entity_average(sequence_output, ctx_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        # pooled_output = self.cls_fc_layer(pooled_output)
        ctx_h = self.ctx_fc_layer(ctx_h)
        e1_h = self.entity_fc_layer1(e1_h)
        e2_h = self.entity_fc_layer2(e2_h)
        si_h = self.si_fc_layer(si_h)

        # Concat -> fc_layer
        concat_h = torch.cat([ctx_h, si_h, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        # print(outputs)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs
        return outputs


class SkeletonAwareRoberta(RobertaPreTrainedModel):
    def __init__(self, config):
        super(SkeletonAwareRoberta, self).__init__(config)
        self.roberta = RobertaModel(config=config)  # Load pretrained model

        self.num_labels = config.num_labels

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.entity_fc_layer1 = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.entity_fc_layer2 = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.si_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        # self.ctx_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)

        self.dense = FCLayer(
            config.hidden_size * 4, config.num_labels, 0.1, use_activation=False
        )

        self.label_classifier = FCLayer(
            # config.hidden_size,
            config.num_labels,
            config.num_labels,
            0.1,
            use_activation=False,
        )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        e1_mask=None,
        e2_mask=None,
        si_mask=None,
        # ctx_mask=None,
    ):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        # print(sequence_output)
        # print(pooled_output)
        # print(sequence_output)
        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        si_h = self.entity_average(sequence_output, si_mask)
        # ctx_h = self.entity_average(sequence_output, ctx_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        # pooled_output = self.cls_fc_layer(pooled_output)
        # ctx_h = self.ctx_fc_layer(ctx_h)
        ctx_h = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer1(e1_h)
        e2_h = self.entity_fc_layer2(e2_h)
        si_h = self.si_fc_layer(si_h)

        # Concat -> fc_layer
        concat_h = torch.cat([ctx_h, si_h, e1_h, e2_h], dim=-1)
        logits = self.dense(concat_h)
        logits = self.label_classifier(logits)
        # print(outputs)

        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        # print(outputs)
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_type = "focal"
                beta = 0.9999
                gamma = 2.0

                loss_fct = CB_loss(beta=beta, gamma=gamma)
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1), loss_type
                )

            outputs = (loss,) + outputs
        return outputs
        # # Softmax
        # return {"loss": loss, "logits": logits}

        # Softmax
        # if labels is not None:
        #     if self.num_labels == 1:
        #         loss_fct = nn.MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = nn.CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #     outputs = (loss,) + outputs
        # return outputs
        # return {"loss": loss, "logits": logits}
