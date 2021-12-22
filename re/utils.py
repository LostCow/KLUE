# coding=utf-8

import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Union
import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
    """

    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    e1_mask: Optional[List[int]] = None
    e2_mask: Optional[List[int]] = None
    si_mask: Optional[List[int]] = None
    ctx_mask: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation",
        "org:dissolved",
        "org:founded",
        "org:place_of_headquarters",
        "org:alternate_names",
        "org:member_of",
        "org:members",
        "org:political/religious_affiliation",
        "org:product",
        "org:founded_by",
        "org:top_members/employees",
        "org:number_of_employees/members",
        "per:date_of_birth",
        "per:date_of_death",
        "per:place_of_birth",
        "per:place_of_death",
        "per:place_of_residence",
        "per:origin",
        "per:employee_of",
        "per:schools_attended",
        "per:alternate_names",
        "per:parents",
        "per:children",
        "per:siblings",
        "per:spouse",
        "per:other_family",
        "per:colleagues",
        "per:product",
        "per:religion",
        "per:title",
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return (
        sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices)
        * 100.0
    )


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c
        )
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions
    # print(probs)

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class RelationExtractionDataset(Dataset):
    """
    A dataset class for loading Relation Extraction data
    """

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = torch.tensor(self.data[idx].input_ids)
        item["attention_mask"] = torch.tensor(self.data[idx].attention_mask)
        item["token_type_ids"] = torch.tensor(self.data[idx].token_type_ids)
        item["e1_mask"] = torch.tensor(self.data[idx].e1_mask)
        item["e2_mask"] = torch.tensor(self.data[idx].e2_mask)
        # item["si_mask"] = torch.tensor(self.data[idx].si_mask)
        item["labels"] = torch.tensor(self.data[idx].label)

        return item

    def __len__(self):
        return len(self.data)


class SKRelationExtractionDataset(Dataset):
    """
    A dataset class for loading Relation Extraction data
    """

    def __init__(self, data):
        self.data = data
        # self.label_counter = self._get_label_counter(self.data)

    def __getitem__(self, idx):
        item = {}
        item["input_ids"] = torch.tensor(self.data[idx].input_ids)
        item["attention_mask"] = torch.tensor(self.data[idx].attention_mask)
        item["token_type_ids"] = torch.tensor(self.data[idx].token_type_ids)
        item["e1_mask"] = torch.tensor(self.data[idx].e1_mask)
        item["e2_mask"] = torch.tensor(self.data[idx].e2_mask)
        item["si_mask"] = torch.tensor(self.data[idx].si_mask)
        # item["ctx_mask"] = torch.tensor(self.data[idx].ctx_mask)
        item["labels"] = torch.tensor(self.data[idx].label)

        return item

    def __len__(self):
        return len(self.data)

    # def get_n_per_labels(self):
    #     return [self.label_counter[i] for i in range(30)]

    # def _get_label_counter(self, data):
    #     all_labels = []
    #     for i in range(len(data)):
    #         all_labels.append(self.data[i].label)
    #     label_counter = Counter(all_labels)
    #     return label_counter
