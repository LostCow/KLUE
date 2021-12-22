import os
import argparse

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
)

from model import BertForSequenceClassification, RobertaForSequenceClassification
from utils import read_json, seed_everything
from dataset import YnatDatasetForTrainer, YnatSoftLabelDatasetForTrainer
from metric import compute_metrics, compute_metrics_for_smoothing


def main(args):
    # setting random seed 42
    # seed_everything()

    # get data path
    train_data_path = os.path.join(args.data_dir, args.train_filename)
    aug_data_path = os.path.join(args.data_dir, args.aug_data_filename)
    valid_data_path = os.path.join(args.data_dir, args.valid_filename)

    # load data
    train_data = read_json(train_data_path)
    aug_data = read_json(aug_data_path)
    valid_data = read_json(valid_data_path)
    total_train_data = train_data + aug_data

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # make dataset
    train_dataset = YnatDatasetForTrainer(data=train_data, tokenizer=tokenizer)
    valid_dataset = YnatDatasetForTrainer(data=valid_data, tokenizer=tokenizer)
    # train_dataset = YnatSoftLabelDatasetForTrainer(data=train_data, tokenizer=tokenizer)
    # valid_dataset = YnatDatasetForTrainer(data=valid_data, tokenizer=tokenizer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 7
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config
    )
    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        seed=args.seed,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # warmup_steps=500,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        logging_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,
        metric_for_best_model="eval_f1",
        # fp16=True,
        # fp16_opt_level="O1",
        eval_steps=args.save_steps,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data_arg
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--model_name_or_path", type=str, default="klue/roberta-large")
    parser.add_argument(
        "--train_filename", type=str, default="ynat-v1.1_train.json"
    )  # ynat-v1.1_dev_sample_10.json
    parser.add_argument(
        "--aug_data_filename", type=str, default="ynat-v1.1_aug_data.json"
    )
    parser.add_argument("--valid_filename", type=str, default="ynat-v1.1_dev.json")

    # train_arg
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.00006755)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.0004648)

    # eval_arg
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)

    args = parser.parse_args()
    main(args)
