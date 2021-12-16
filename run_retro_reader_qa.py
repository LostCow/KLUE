import logging
import os
import sys
import datasets
from datasets import Dataset
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from arguments import DataTrainingArguments, SketchModelArguments, IntensiveModelArguments
from utils import create_pandas, read_json
from retro_reader import RetroReader

import wandb


logger = logging.getLogger(__name__)


def setup_logging(training_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def convert_json_to_dataset(dataset_path="", train_file_name="", validation_file_name=""):
    train_data_path = os.path.join(dataset_path, train_file_name)
    valid_data_path = os.path.join(dataset_path, validation_file_name)
    train_df = create_pandas(read_json(train_data_path))
    valid_df = create_pandas(read_json(valid_data_path))
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    return train_dataset, valid_dataset


def main():
    parser = HfArgumentParser(
        (SketchModelArguments, IntensiveModelArguments, DataTrainingArguments, TrainingArguments)
    )

    (
        sketch_model_args,
        intensive_model_args,
        data_args,
        base_training_args,
    ) = parser.parse_args_into_dataclasses()
    setup_logging(base_training_args)
    set_seed(base_training_args.seed)

    # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)

    train_examples, valid_examples = convert_json_to_dataset(
        dataset_path=data_args.dataset_path,
        train_file_name=data_args.train_file,
        validation_file_name=data_args.validation_file,
    )

    # data_collator = (
    #     default_data_collator
    #     if data_args.pad_to_max_length
    #     else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if base_training_args.fp16 else None)
    # )
    trainer = RetroReader(
        sketch_model_name_or_path=sketch_model_args.sketch_model_name_or_path,
        intensive_model_name_or_path=intensive_model_args.intensive_model_name_or_path,
        base_training_args=base_training_args,
        data_args=data_args,
        train_examples=train_examples,
        eval_examples=valid_examples,
        # tokenizer=tokenizer,
        # data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"

    main()
