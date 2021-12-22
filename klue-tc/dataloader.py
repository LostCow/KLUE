import torch
from dataset import YnatDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from utils import read_json


class YnatDataLoader(object):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length else self.tokenizer.model_max_length

    def collate_fn(self, input_examples):
        input_texts, input_labels = [], []
        for input_example in input_examples:
            text, label = input_example
            input_texts.append(text)
            input_labels.append(label)

        encoded_texts = self.tokenizer.batch_encode_plus(
            input_texts,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=True,
            return_attention_mask=True,
        )  # input_ids, token_type_ids, attention_mask
        input_ids = encoded_texts["input_ids"]
        token_type_ids = encoded_texts["token_type_ids"]
        attention_mask = encoded_texts["attention_mask"]
        return input_ids, token_type_ids, attention_mask, torch.tensor(input_labels)

    def get_dataloader(self, file_path, batch_size, **kwargs):
        data = read_json(file_path)
        dataset = YnatDataset(data)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            **kwargs
        )
