import json
import logging
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class KlueStsInputExample:
    """A single training/test example for klue semantic textual similarity.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        score: float. The label of the example.
        binary_label: int. 0: False, 1: True
    """

    guid: str
    text_a: str
    text_b: str
    label: float
    binary_label: int

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


@dataclass(frozen=True)
class KlueStsInputFeatures:
    """A single set of features of data. Property names are the same names as the corresponding inputs to a model.

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
    s1_mask: Optional[List[int]] = None
    s2_mask: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class KlueStsDataset:
    def __init__(self, data: list, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        """Dataset for KlueStsDataset

        Args:
            data: json-loaded list
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.features = self._convert_features(self._create_examples(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.features[idx]
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attn_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(
            0 if feature.token_type_ids is None else feature.token_type_ids,
            dtype=torch.long,
        )
        labels = torch.tensor(feature.label, dtype=torch.float)
        return (input_ids, attn_mask, token_type_ids, labels)

    def _create_examples(self, data):
        examples = [
            KlueStsInputExample(
                guid=d["guid"],
                text_a=d["sentence1"],
                text_b=d["sentence2"],
                label=d["labels"]["real-label"],
                binary_label=d["labels"]["binary-label"],
            )
            for d in self.data
        ]
        return examples

    def _convert_features(
        self, examples: List[KlueStsInputExample]
    ) -> List[KlueStsInputFeatures]:
        return convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=self.max_seq_length,
            label_list=[],
            output_mode="regression",
        )


class KlueStsWithSentenceMaskDataset:
    def __init__(self, data: list, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        """Dataset for KlueStsDataset

        Args:
            data: json-loaded list
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.features = self._convert_features(self._create_examples(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.features[idx]
        item = {}
        item["input_ids"] = torch.tensor(feature.input_ids, dtype=torch.long)
        item["attention_mask"] = torch.tensor(feature.attention_mask, dtype=torch.long)
        item["token_type_ids"] = torch.tensor(
            0 if feature.token_type_ids is None else feature.token_type_ids,
            dtype=torch.long,
        )
        item["s1_mask"] = torch.tensor(feature.s1_mask, dtype=torch.long)
        item["s2_mask"] = torch.tensor(feature.s2_mask, dtype=torch.long)
        item["labels"] = torch.tensor(feature.label, dtype=torch.float)
        return item

    def _create_examples(self, data):
        examples = [
            KlueStsInputExample(
                guid=d["guid"],
                text_a=d["sentence1"],
                text_b=d["sentence2"],
                label=d["labels"]["real-label"],
                binary_label=d["labels"]["binary-label"],
            )
            for d in data
        ]
        return examples

    def _convert_features(
        self, examples: List[KlueStsInputExample]
    ) -> List[KlueStsInputFeatures]:
        return convert_examples_to_features_with_sen_embed(
            examples,
            self.tokenizer,
            max_length=self.max_seq_length,
            label_list=[],
            output_mode="regression",
        )


def convert_examples_to_features(
    examples: List[KlueStsInputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.model_max_length

    labels = [float(example.label) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = KlueStsInputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


def convert_examples_to_features_with_sen_embed(
    examples: List[KlueStsInputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.model_max_length

    labels = [float(example.label) for example in examples]

    batch_encoding = tokenizer(
        [
            (
                example.text_a,
                example.text_b,
            )
            for example in examples
        ],
        max_length=max_length,
        padding=True,
        truncation=True,
    )
    tatal_s1_mask = []
    tatal_s2_mask = []
    for i in range(len(examples)):
        s1_mask = [0] * len(batch_encoding["input_ids"][i])
        s2_mask = [0] * len(batch_encoding["input_ids"][i])
        # print(batch_encoding["token_type_ids"][i])

        idx = 1  # cls 토큰 다음부터
        while batch_encoding["input_ids"][i][idx] != tokenizer.sep_token_id:
            s1_mask[idx] = 1
            idx += 1
        idx += 1  # sep 토큰 다음부터
        while batch_encoding["input_ids"][i][idx] != tokenizer.sep_token_id:
            s2_mask[idx] = 1
            idx += 1
        tatal_s1_mask.append(s1_mask)
        tatal_s2_mask.append(s2_mask)
    batch_encoding["s1_mask"] = tatal_s1_mask
    batch_encoding["s2_mask"] = tatal_s2_mask

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = KlueStsInputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features
