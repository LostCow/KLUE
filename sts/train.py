import argparse
import numpy as np

import os
import torch
from transformers import AutoTokenizer, AutoConfig, Trainer, TrainingArguments

from model import RobertaForStsRegression
from dataset import KlueStsWithSentenceMaskDataset
from utils import read_json, seed_everything
from metric import compute_metrics


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = args.num_labels
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_file_path = os.path.join(args.data_dir, args.train_filename)
    valid_file_path = os.path.join(args.data_dir, args.valid_filename)

    train_json = read_json(train_file_path)
    valid_json = read_json(valid_file_path)

    train_dataset = KlueStsWithSentenceMaskDataset(train_json, tokenizer, 510)
    valid_dataset = KlueStsWithSentenceMaskDataset(train_json, tokenizer, 510)

    model = RobertaForStsRegression.from_pretrained(
        args.model_name_or_path, config=config
    )
    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        logging_dir="./logs",
        logging_steps=args.save_steps,
        evaluation_strategy=args.evaluation_strategy,
        metric_for_best_model="pearsonr",
        fp16=True,
        fp16_opt_level="O1",
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
        "--train_filename", type=str, default="klue-sts-v1.1_train.json"
    )
    parser.add_argument("--valid_filename", type=str, default="klue-sts-v1.1_dev.json")

    # train_arg
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # eval_arg
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_total_limit", type=int, default=2)

    args = parser.parse_args()
    main(args)
