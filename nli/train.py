import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers.trainer_utils import EvalPrediction
import datasets
import os
import random
import numpy as np

import pandas as pd


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def create_pandas(data: dict) -> pd.DataFrame:
    guid = []
    source = []
    premise = []
    hypothesis = []
    label = []
    for entry in data:
        guid.append(entry["guid"])
        if "genre" in entry.keys():
            source.append(entry["genre"])
        else:
            source.append(entry["source"])
        premise.append(entry["premise"])
        hypothesis.append(entry["hypothesis"])
        if entry["gold_label"] == "entailment":
            label.append(0)
        elif entry["gold_label"] == "contradiction":
            label.append(1)
        elif entry["gold_label"] == "neutral":
            label.append(2)

    df = pd.DataFrame(
        data={
            "guid": guid,
            "source": source,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
        }
    )
    return df


def main():
    model_name_or_path = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    data_dir = "./klue-nli-v1.1"
    train_filename = "klue-nli-v1.1_train.json"
    valid_filename = "klue-nli-v1.1_dev.json"

    train_file_path = os.path.join(data_dir, train_filename)
    valid_file_path = os.path.join(data_dir, valid_filename)

    train_json = read_json(train_file_path)
    valid_json = read_json(valid_file_path)

    train_df = create_pandas(train_json)
    valid_df = create_pandas(valid_json)
    train_dataset = datasets.Dataset.from_pandas(train_df)
    valid_dataset = datasets.Dataset.from_pandas(valid_df)

    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            padding="max_length",
            return_token_type_ids=False,
            max_length=128,
        )

    train_examples = train_dataset.map(preprocess_function, batched=True)
    valid_examples = valid_dataset.map(preprocess_function, batched=True)

    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=3)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    accuracy = datasets.load_metric("glue", "qnli").compute

    def compute_metrics(p: EvalPrediction):
        predictions = p.predictions.argmax(axis=1)
        references = p.label_ids
        metric = accuracy(predictions=predictions, references=references)
        return metric

    training_args = TrainingArguments(
        output_dir="outputs",
        save_total_limit=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=10,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        weight_decay=0,
        # fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=valid_examples,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    main()
