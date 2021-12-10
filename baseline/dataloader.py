import torch
from dataset import KlueMrcDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from utils import read_json


class KlueMrcDataLoaderGetter(object):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = None,
        doc_stride: int = None,
        max_query_length: int = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = (
            max_seq_length if max_seq_length else self.tokenizer.model_max_length
        )
        assert doc_stride is not None
        assert max_query_length is not None
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length

    def collate_fn(self, input_examples):
        """KlueMrcFeature padded all input up to max_seq_length"""
        pass

    def get_dataloader(self, file_path, batch_size, **kwargs):
        data = read_json(file_path)
        dataset = KlueMrcDataset(
            data,
            self.tokenizer,
            self.max_seq_length,
            self.doc_stride,
            self.max_query_length,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
