import torch

from transformers import Trainer
import numpy as np

from losses import LDAMLoss


class LDAMLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_per_labels = self.train_dataset.get_n_per_labels()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        betas = [0, 0.99]
        beta_idx = self.state.epoch >= 2
        n_per_labels = self.n_per_labels

        effective_num = 1.0 - np.power(betas[beta_idx], n_per_labels)
        cls_weights = (1.0 - betas[beta_idx]) / np.array(effective_num)
        cls_weights = cls_weights / np.sum(cls_weights) * len(n_per_labels)
        cls_weights = torch.FloatTensor(cls_weights)

        criterion = LDAMLoss(
            cls_num_list=n_per_labels, max_m=0.5, s=30, weight=cls_weights
        )
        if torch.cuda.is_available():
            criterion.cuda()

        loss_fct = criterion(logits, labels)
        return (loss_fct, outputs) if return_outputs else loss_fct
