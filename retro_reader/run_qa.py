import logging
import os
import sys
import datasets
from datasets import Dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from arguments import DataTrainingArguments, ModelArguments
from trainer_qa import QuestionAnsweringTrainer
from utils_qa import create_pandas, read_json
from processor import DataProcessor
from metric import compute_metrics
from retro_reader import RetroReader

import wandb


logger = logging.getLogger(__name__)


def setup_logging(training_args):
    # Setup logging
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
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def main():
    # wandb.init()
    # parser setting, we don't use argments to json
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    setup_logging(training_args)
    set_seed(training_args.seed)

    # config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=True
    )

    # model = AutoModelForQuestionAnswering.from_pretrained(model_args.model_name_or_path, config=config)

    train_data_path = os.path.join(data_args.dataset_path, data_args.train_file)
    valid_data_path = os.path.join(data_args.dataset_path, data_args.validation_file)
    train_df = create_pandas(read_json(train_data_path)).iloc[:100]
    valid_df = create_pandas(read_json(valid_data_path)).iloc[:100]
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )
    trainer = RetroReader(
        # model=model,
        model_name_or_path=model_args.model_name_or_path,
        training_args=training_args,
        data_args=data_args,
        train_examples=train_dataset if training_args.do_train else None,
        eval_examples=valid_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"
    main()
