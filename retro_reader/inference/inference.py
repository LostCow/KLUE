import argparse
import os
import tarfile

import torch
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    DataCollatorWithPadding,
)
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.processors.squad import SquadResult

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from typing import List
import datasets
from torch.utils.data import Dataset
from transformers.data.processors.squad import SquadExample, squad_convert_examples_to_features
from transformers.data.processors.utils import InputFeatures
import json
import collections
from arguments import DataTrainingArguments, ModelArguments, MyTrainingArguments

from retro_reader import RetroReader
from model import RobertaForSequenceClassification, RobertaForQuestionAnsweringAVPool
from utils_qa import create_pandas

KLUE_MRC_OUTPUT = "output.csv"  # the name of output file should be output.csv


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


def get_score1():
    cof = [1, 1]
    best_cof = [1]
    all_scores = collections.OrderedDict()
    idx = 0
    for input_file in "sketch_reader_outputs/cls_score.json,intensive_reader_outputs/null_odds.json".split(
        ","
    ):
        with open(input_file, "r") as reader:
            input_data = json.load(reader, strict=False)
            for (key, score) in input_data.items():
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(cof[idx] * score)
        idx += 1
    output_scores = {}
    for (key, scores) in all_scores.items():
        mean_score = 0.0
        for score in scores:
            mean_score += score
        mean_score /= float(len(scores))
        output_scores[key] = mean_score

    idx = 0
    all_nbest = collections.OrderedDict()
    for input_file in "intensive_reader_outputs/nbest_predictions.json".split(","):
        with open(input_file, "r") as reader:
            input_data = json.load(reader, strict=False)
            for (key, entries) in input_data.items():
                if key not in all_nbest:
                    all_nbest[key] = collections.defaultdict(float)
                for entry in entries:
                    all_nbest[key][entry["text"]] += best_cof[idx] * entry["probability"]
        idx += 1
    output_predictions = {}
    for (key, entry_map) in all_nbest.items():
        sorted_texts = sorted(entry_map.keys(), key=lambda x: entry_map[x], reverse=True)
        best_text = sorted_texts[0]
        output_predictions[key] = best_text

    best_th = 0

    for qid in output_predictions.keys():
        if output_scores[qid] > best_th:
            output_predictions[qid] = ""

    output_prediction_file = KLUE_MRC_OUTPUT
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(output_predictions, indent=4) + "\n")


def load_model_and_type(model_dir: str):
    """load model and model type from tar file pre-fetched from s3

    Args:
        model_dir: str: the directory of tar file
        model_tar_path: str: the name of tar file
    """

    sketch_tarpath = os.path.join(model_dir, "sketch.tar.gz")
    intensive_tarpath = os.path.join(model_dir, "intensive.tar.gz")

    tar1 = tarfile.open(sketch_tarpath, "r:gz")
    tar2 = tarfile.open(intensive_tarpath, "r:gz")
    tar1.extractall(path=os.path.join(model_dir, "sketch_reader"))
    tar2.extractall(path=os.path.join(model_dir, "intensive_reader"))


@torch.no_grad()
def inference() -> None:
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_dir = "./model"

    load_model_and_type(model_dir=model_dir)

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

    valid_data_path = os.path.join(os.environ.get("SM_CHANNEL_EVAL", "/data"), "klue-mrc-v1.1_test.json")
    valid_df = create_pandas(read_json(valid_data_path))
    valid_dataset = datasets.Dataset.from_pandas(valid_df)

    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )
    trainer = RetroReader(
        sketch_model_name_or_path=os.path.join(model_dir, "sketch_reader"),
        intensive_model_name_or_path=os.path.join(model_dir, "intensive_reader"),
        training_args=training_args,
        data_args=data_args,
        eval_examples=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.sketch_reader.predict(
        eval_dataset=trainer.eval_dataset_for_sketch_reader,
        eval_examples=trainer.eval_examples,
    )
    trainer.intensive_reader.predict(
        eval_dataset=trainer.eval_dataset_for_intensive_reader,
        eval_examples=trainer.eval_examples,
    )

    get_score1()


if __name__ == "__main__":
    os.environ["WANDB_DISABLED"] = "true"

    inference()
