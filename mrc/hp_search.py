import logging
import os
import sys
from torch import nn
import numpy as np
from torch.nn.modules import module
import datasets
from datasets.load import load_metric
from datasets import Dataset
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    AutoConfig,
    default_data_collator,
    DataCollatorWithPadding,
    AutoTokenizer,
)
from processor import DataProcessor
from transformers.trainer_utils import (
    EvalLoopOutput,
    PredictionOutput,
    EvalPrediction,
    speed_metrics,
    denumpify_detensorize,
)
from arguments import DataTrainingArguments, SketchModelArguments, IntensiveModelArguments
from utils import convert_json_to_dataset
from retro_reader import RetroReader, IntensiveReader
from model import RobertaCNNForQuestionAnsweringAVPool

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

    train_examples, valid_examples = convert_json_to_dataset(
        dataset_path=data_args.dataset_path,
        train_file_name=data_args.train_file,
        validation_file_name=data_args.validation_file,
    )

    intensive_tokenizer = AutoTokenizer.from_pretrained(
        intensive_model_args.intensive_model_name_or_path, use_fast=True
    )
    intensive_data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(
            intensive_tokenizer, pad_to_multiple_of=8 if base_training_args.fp16 else None
        )
    )
    column_names = train_examples.column_names
    mrc_processor = DataProcessor(
        data_args=data_args,
        sketch_tokenizer=intensive_tokenizer,
        intensive_tokenizer=intensive_tokenizer,
        column_names=column_names,
    )

    def preprocess_examples(module_name="sketch"):
        with base_training_args.main_process_first(
            desc=f"train dataset for {module_name} reader map pre-processing"
        ):
            train_dataset = train_examples.map(
                mrc_processor.prepare_train_features_for_intensive_reader,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Running tokenizer on train dataset for {module_name} reader",
            )

        with base_training_args.main_process_first(
            desc=f"validation dataset for {module_name} reader map pre-processing"
        ):
            eval_dataset = valid_examples.map(
                mrc_processor.prepare_eval_features_for_intensive_reader,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Running tokenizer on validation dataset for {module_name} reader",
            )
        return train_dataset, eval_dataset

    intensive_reader_config = AutoConfig.from_pretrained(intensive_model_args.intensive_model_name_or_path)

    def model_init():
        return RobertaCNNForQuestionAnsweringAVPool.from_pretrained(
            "klue/roberta-large", config=intensive_reader_config
        )

    (
        train_dataset_for_intensive_reader,
        eval_dataset_for_intensive_reader,
    ) = preprocess_examples(module_name="intensive")

    def compute_metrics(p: EvalPrediction):
        metric = load_metric("squad_v2")
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    intensive_reader = IntensiveReader(
        model_init=model_init,
        args=base_training_args,
        data_args=data_args,
        train_dataset=train_dataset_for_intensive_reader if base_training_args.do_train else None,
        eval_dataset=eval_dataset_for_intensive_reader if base_training_args.do_eval else None,
        eval_examples=valid_examples if base_training_args.do_eval else None,
        tokenizer=intensive_tokenizer,
        data_collator=intensive_data_collator,
        compute_metrics=compute_metrics,
    )

    def my_hp_space(trial):
        return {
            # "learning_rate": trial.suggest_float("learning_rate", 2e-5, 1e-4, log=True),
            "seed": trial.suggest_int("seed", 1, 123),
            # "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 24, 32, 45]),
            # "weight_decay": trial.suggest_float("weight_decay", 0, 0.3),
            # "gradient_accumulation_steps": trial.suggest_categorical(
            #     "gradient_accumulation_steps", [8, 16, 32, 64]
            # ),
        }

    intensive_reader.hyperparameter_search(
        direction="maximize",
        hp_space=my_hp_space,
    )


if __name__ == "__main__":
    # os.environ["WANDB_DISABLED"] = "true"

    main()
