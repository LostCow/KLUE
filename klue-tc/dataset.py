from typing import List
import torch
from torch.utils.data import Dataset
from collections import Counter


class YnatDataset(Dataset):

    label2idx = {"정치": 0, "경제": 1, "사회": 2, "생활문화": 3, "세계": 4, "IT과학": 5, "스포츠": 6}

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        this_data = self.data[index]
        text = this_data["title"]
        label = this_data["label"]
        label_idx = self.label2idx[label]
        return text, label_idx


class YnatDatasetForTrainer(Dataset):
    label2idx = {"정치": 0, "경제": 1, "사회": 2, "생활문화": 3, "세계": 4, "IT과학": 5, "스포츠": 6}

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.features = self._create_features(data=self.data)
        self.labels = self._create_labels(data=self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.features.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def _create_features(self, data):
        texts = []
        for d in data:
            texts.append(d["title"])
        features = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        return features

    def _create_labels(self, data):
        labels = []
        for d in data:
            labels.append(self.label2idx[d["label"]])
        return labels


class YnatSoftLabelDatasetForTrainer(Dataset):
    label2idx = {
        "정치": 0,
        "경제": 1,
        "사회": 2,
        "생활문화": 3,
        "세계": 4,
        "IT과학": 5,
        "스포츠": 6,
        "해당없음": 7,
    }

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.features = self._create_features(data=self.data)
        self.labels = self._create_labels(data=self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {k: v[index] for k, v in self.features.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item

    def _create_features(self, data):
        texts = []
        for d in data:
            texts.append(d["title"])
        features = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        return features

    def _create_labels(self, data) -> List[List[int]]:
        labels = []
        for d in data:
            label = []
            label_vector = [0] * (len(self.label2idx) - 1)
            for k, v in d["annotations"]["annotations"].items():
                for category in v:
                    label.append(self.label2idx[category])
            count_label = Counter(label)
            if self.label2idx["해당없음"] in count_label:
                count_label.pop(self.label2idx["해당없음"])

            total_value_sum = sum(count_label.values())

            for label_idx in count_label.keys():
                label_vector[label_idx] = count_label[label_idx] / total_value_sum
            labels.append(label_vector)
        return labels
