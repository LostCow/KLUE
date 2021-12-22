import os
import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
)

from model import RobertaForClassificationWithMasking
from utils import read_json
from dataset import KlueNliWithSentenceMaskDataset
from metric import compute_metrics
import wandb

model_name = "klue/roberta-large"
wandb.init(
    project="klue-benchmark_hp_search",
    name="roberta",
)


def my_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 4, 5),
        "seed": trial.suggest_int("seed", 1, 42),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [32, 64]
        ),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [1, 2, 4, 8]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1),
    }


def model_init():
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = 3
    model = RobertaForClassificationWithMasking.from_pretrained(
        model_name, config=config
    )
    return model


def my_objective(metrics):
    # Your elaborate computation here
    return metrics["eval_accuracy"]


# load data
train_data = read_json("/opt/ml/KLUE/klue-nli/data/klue-nli-v1.1_train.json")
# aug_data = read_json(aug_data_path)
valid_data = read_json("/opt/ml/KLUE/klue-nli/data/klue-nli-v1.1_dev.json")
# total_train_data = train_data + aug_data

tokenizer = AutoTokenizer.from_pretrained(model_name)

# make dataset
train_dataset = KlueNliWithSentenceMaskDataset(
    data=train_data, tokenizer=tokenizer, max_seq_length=510
)
valid_dataset = KlueNliWithSentenceMaskDataset(
    data=valid_data, tokenizer=tokenizer, max_seq_length=510
)
# train_dataset = YnatSoftLabelDatasetForTrainer(data=train_data, tokenizer=tokenizer)
# valid_dataset = YnatDatasetForTrainer(data=valid_data, tokenizer=tokenizer)


training_args = TrainingArguments(
    "hp_search_roberta",
    save_total_limit=1,
    evaluation_strategy="steps",
    eval_steps=100,
    metric_for_best_model="eval_accuracy",
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

trainer.hyperparameter_search(
    direction="maximize", hp_space=my_hp_space, compute_objective=my_objective
)

# model.save_pretrained(args.model_dir)
# tokenizer.save_pretrained(args.model_dir)
