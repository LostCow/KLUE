import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class KlueNliInputExample:
    """A single training/test example for klue natural language inference.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        label: The label of the example.
    """

    guid: str
    text_a: str
    text_b: str
    label: float

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


@dataclass(frozen=True)
class KlueNliInputFeatures:
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
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class KlueNliDataset:
    labels = ["entailment", "contradiction", "neutral"]

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
            KlueNliInputExample(
                guid=d["guid"],
                text_a=d["premise"],
                text_b=d["hypothesis"],
                label=d["gold_label"],
            )
            for d in self.data
        ]
        return examples

    def _convert_features(
        self, examples: List[KlueNliInputExample]
    ) -> List[KlueNliInputFeatures]:
        return convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=self.max_seq_length,
            label_list=self.labels,
        )


def convert_examples_to_features(
    examples: List[KlueNliInputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    label_list=None,
):
    if max_length is None:
        max_length = tokenizer.model_max_length

    label_map = {label: i for i, label in enumerate(label_list)}
    labels = [label_map[example.label] for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = KlueNliInputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features
