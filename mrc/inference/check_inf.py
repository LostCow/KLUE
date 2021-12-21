import argparse
import os
import tarfile

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
from transformers.data.metrics.squad_metrics import compute_predictions_logits
from transformers.data.processors.squad import SquadResult

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from typing import List

import torch
from torch.utils.data import Dataset
from transformers.data.processors.squad import SquadExample, squad_convert_examples_to_features
from transformers.data.processors.utils import InputFeatures
import json


def read_json(file_path):
    with open(file_path) as f:
        return json.load(f)


class KlueMrcExample(SquadExample):
    def __init__(self, question_type: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.question_type = question_type


class KlueMrcDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_seq_length,
        doc_stride,
        max_query_length,
        convert_examples=True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.examples = self._create_examples(data)
        if convert_examples:
            self.features = self._convert_to_features(self.examples)
        else:
            self.features = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attn_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(
            0 if feature.token_type_ids is None else feature.token_type_ids,
            dtype=torch.long,
        )
        return input_ids, attn_mask, token_type_ids, idx

    def _create_examples(self, data) -> List[KlueMrcExample]:
        examples = list()
        for entry in data["data"]:
            for paragraph in entry["paragraphs"]:
                for qa in paragraph["qas"]:
                    is_impossible = qa.get("is_impossible", False)

                    answers = qa["answers"]
                    answer_text = None
                    start_position_character = None

                    example = KlueMrcExample(
                        question_type=qa.get("question_type", 1),
                        qas_id=qa["guid"],
                        question_text=qa["question"],
                        context_text=paragraph["context"],
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=entry["title"],
                        answers=answers,
                        is_impossible=is_impossible,
                    )
                    examples.append(example)
        return examples

    def _convert_to_features(self, examples: List[KlueMrcExample]) -> List[InputFeatures]:
        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=False,
            return_dataset=False,
            threads=10,
        )
        return features


class KlueMrcDataLoaderGetter(object):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = None,
        doc_stride: int = None,
        max_query_length: int = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length if max_seq_length else self.tokenizer.model_max_length
        assert doc_stride is not None
        assert max_query_length is not None
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length

    def collate_fn(self, input_examples):
        """KlueMrcFeature padded all input up to max_seq_length"""
        pass

    def get_dataloader(self, file_path, batch_size, **kwargs):
        data = read_json(file_path)
        dataset = KlueMrcDataset(
            data,
            self.tokenizer,
            self.max_seq_length,
            self.doc_stride,
            self.max_query_length,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)


KLUE_MRC_OUTPUT = "output.csv"  # the name of output file should be output.csv


def load_model_and_type(model_dir: str, model_tar_file: str):
    """load model and model type from tar file pre-fetched from s3

    Args:
        model_dir: str: the directory of tar file
        model_tar_path: str: the name of tar file
    """
    # tarpath = os.path.join(model_dir, model_tar_file)
    # tar = tarfile.open(tarpath, "r:gz")
    # tar.extractall(path=model_dir)

    model = AutoModelForQuestionAnswering.from_pretrained("klue/roberta-large")
    config = AutoConfig.from_pretrained("klue/roberta-large")
    return model, config.model_type


@torch.no_grad()
def inference(data_dir: str, model_dir: str, output_dir: str, args) -> None:
    # configure gpu
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model, model_type = load_model_and_type(model_dir, args.model_tar_file)
    model.to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.eval()
    kwargs = {"num_workers": num_gpus, "pin_memory": True} if torch.cuda.is_available() else {}
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large", use_fast=False)
    # infer
    # output_file_path = os.path.join(output_dir, KLUE_MRC_OUTPUT)
    output_file_path = os.path.join("rrrrrrrrr.csv")

    klue_mrc_dataloader_getter = KlueMrcDataLoaderGetter(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
    )
    klue_mrc_dataloader = klue_mrc_dataloader_getter.get_dataloader(
        # file_path=os.path.join(data_dir, args.test_filename), batch_size=args.batch_size, **kwargs
        file_path="/opt/ml/klue/klue-mrc-v1.1/klue-mrc-v1.1_dev_sample_10.json",
        batch_size=args.batch_size,
        **kwargs
    )
    qa_result = list()
    for data in tqdm(klue_mrc_dataloader):
        input_ids, attention_mask, token_type_ids, idx = data
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)

        # roberta does not accept token_type_id > 1
        if model_type == "roberta":
            token_type_ids = None

        # start_logits, end_logits = model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        # )
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        start_logits, end_logits = output["start_logits"], output["end_logits"]
        batch_results = list()

        for i, feature_index in enumerate(idx):
            unique_id = klue_mrc_dataloader.dataset.features[feature_index].unique_id
            single_example_start_logits = start_logits[i].tolist()
            single_example_end_logits = end_logits[i].tolist()
            batch_results.append(
                SquadResult(unique_id, single_example_start_logits, single_example_end_logits)
            )

        qa_result.extend(batch_results)

    examples = klue_mrc_dataloader.dataset.examples
    features = klue_mrc_dataloader.dataset.features
    do_lower_case = getattr(tokenizer, "do_lower_case", False)

    preds = compute_predictions_logits(
        all_examples=examples,
        all_features=features,
        all_results=qa_result,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        do_lower_case=do_lower_case,
        output_prediction_file=output_file_path,
        output_nbest_file=None,
        output_null_log_odds_file=None,
        verbose_logging=False,
        version_2_with_negative=True,
        null_score_diff_threshold=0,
        tokenizer=tokenizer,
    )
    return preds


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for inference (default: 64)",
    )
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/data"))
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/output"),
    )
    parser.add_argument(
        "--model_tar_file",
        type=str,
        default="klue-mrc.tar.gz",
        help="it needs to include all things for loading baseline model & tokenizer, \
             only supporting transformers.AutoModelForSequenceClassification as a model \
             transformers.XLMRobertaTokenizer or transformers.BertTokenizer as a tokenizer",
    )
    parser.add_argument(
        "--test_filename",
        # default="klue-mrc-v1.1_test.json",
        default="klue-mrc-v1.1_dev.json",
        type=str,
        help="Name of the test file (default: klue-mrc-v1.1_test.json)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded. (default: 510)",
    )
    parser.add_argument(
        "--max_query_length",
        type=int,
        default=64,
        help="the maximum number of tokens for the question (default: 64)",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks, how much stride to take between chunks. (default: 128)",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    inference(data_dir, model_dir, output_dir, args)
