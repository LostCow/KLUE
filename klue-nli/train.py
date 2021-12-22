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


def main(args):
    wandb.init(
        project="klue-benchmark",
        name="roberta",
    )
    # setting random seed 42
    # seed_everything()

    # get data path
    train_data_path = os.path.join(args.data_dir, args.train_filename)
    valid_data_path = os.path.join(args.data_dir, args.valid_filename)

    # load data
    train_data = read_json(train_data_path)
    valid_data = read_json(valid_data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # make dataset
    train_dataset = KlueNliWithSentenceMaskDataset(
        data=train_data, tokenizer=tokenizer, max_seq_length=510
    )
    valid_dataset = KlueNliWithSentenceMaskDataset(
        data=valid_data, tokenizer=tokenizer, max_seq_length=510
    )
    # max_seq_length는 사실 하는게 없음, 수정해야할 것
    # train_dataset = YnatSoftLabelDatasetForTrainer(data=train_data, tokenizer=tokenizer)
    # valid_dataset = YnatDatasetForTrainer(data=valid_data, tokenizer=tokenizer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = args.num_labels
    model = RobertaForClassificationWithMasking.from_pretrained(
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
        metric_for_best_model="eval_accuracy",
        # fp16=True,
        # fp16_opt_level="O1",
        eval_steps=args.save_steps,
        load_best_model_at_end=True,
        report_to="wandb",
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
        "--train_filename", type=str, default="klue-nli-v1.1_train.json"
    )  # ynat-v1.1_dev_sample_10.json
    parser.add_argument("--valid_filename", type=str, default="klue-nli-v1.1_dev.json")

    # train_arg
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=15)
    parser.add_argument("--num_train_epochs", type=int, default=7)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.00006526)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.077)

    # eval_arg
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)

    args = parser.parse_args()
    main(args)
