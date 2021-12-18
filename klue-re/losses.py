import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def focal_loss(labels, logits, alpha, gamma):
    BCLoss = F.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none"
    )

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(
            -gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits))
        )

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class CB_loss(nn.Module):
    def __init__(self, beta, gamma, epsilon=0.1):
        super(CB_loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits, labels, loss_type="softmax"):
        # self.epsilon = 0.1 #labelsmooth
        beta = self.beta
        gamma = self.gamma

        no_of_classes = logits.shape[1]
        samples_per_cls = torch.Tensor(
            [sum(labels == i) for i in range(logits.shape[1])]
        )
        if torch.cuda.is_available():
            samples_per_cls = samples_per_cls.cuda()

        effective_num = 1.0 - torch.pow(beta, samples_per_cls)
        weights = (1.0 - beta) / ((effective_num) + 1e-8)

        weights = weights / torch.sum(weights) * no_of_classes
        labels = labels.reshape(-1, 1)

        weights = torch.tensor(weights.clone().detach()).float()

        if torch.cuda.is_available():
            weights = weights.cuda()
            labels_one_hot = (
                torch.zeros(len(labels), no_of_classes)
                .cuda()
                .scatter_(1, labels, 1)
                .cuda()
            )

        labels_one_hot = (
            1 - self.epsilon
        ) * labels_one_hot + self.epsilon / no_of_classes
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, no_of_classes)

        if loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(
                input=logits, target=labels_one_hot, pos_weight=weights
            )
        elif loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(
                input=pred, target=labels_one_hot, weight=weights
            )
        return cb_loss


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.bool)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(
            self.s * output.to("cuda"), target.to("cuda"), weight=self.weight.to("cuda")
        )
