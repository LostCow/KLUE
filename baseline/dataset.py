from typing import List

import torch
from torch.utils.data import Dataset
from transformers.data.processors.squad import (
    SquadExample, squad_convert_examples_to_features)
from transformers.data.processors.utils import InputFeatures


class KlueMrcExample(SquadExample):
    def __init__(self, question_type: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.question_type = question_type


class KlueMrcDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        convert_examples=True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.examples = self._create_examples(data)
        if convert_examples:
            self.features = self._convert_to_features(self.examples)
        else:
            self.features = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attn_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(
            0 if feature.token_type_ids is None else feature.token_type_ids,
            dtype=torch.long,
        )
        return input_ids, attn_mask, token_type_ids, idx

    def _create_examples(self, data) -> List[KlueMrcExample]:
        examples = list()
        for entry in data["data"]:
            for paragraph in entry["paragraphs"]:
                for qa in paragraph["qas"]:
                    is_impossible = qa.get("is_impossible", False)

                    answers = qa["answers"]
                    answer_text = None
                    start_position_character = None

                    example = KlueMrcExample(
                        question_type=qa.get("question_type", 1),
                        qas_id=qa["guid"],
                        question_text=qa["question"],
                        context_text=paragraph["context"],
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=entry["title"],
                        answers=answers,
                        is_impossible=is_impossible,
                    )
                    examples.append(example)
        return examples

    def _convert_to_features(
        self, examples: List[KlueMrcExample]
    ) -> List[InputFeatures]:
        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=False,
            return_dataset=False,
            threads=10,
        )
        return features
