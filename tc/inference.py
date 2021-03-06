""" Usage
$ python inference.py --data_dir /data \
                      --model_dir /model \
                      --output_dir /output \
                      [args ...]
"""
import argparse
import os
import tarfile

import torch
from dataloader import YnatDataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from model import BertForSequenceClassification

YNAT_OUTPUT = "output.csv"


def load_model(saved_model, filename):
    tarpath = os.path.join(saved_model, filename)
    tar = tarfile.open(tarpath, "r:gz")
    tar.extractall(path=saved_model)

    model = BertForSequenceClassification.from_pretrained(saved_model)

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    num_gpus = torch.cuda.device_count()
    use_cuda = num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_dir, args.model_tar_file).to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    kwargs = {"num_workers": num_gpus, "pin_memory": True} if use_cuda else {}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    ynat_data_loader = YnatDataLoader(tokenizer, args.max_length)
    ynat_test_loader = ynat_data_loader.get_dataloader(
        os.path.join(data_dir, args.test_filename),
        batch_size=args.batch_size,
        **kwargs,
    )

    output_file = open(os.path.join(output_dir, YNAT_OUTPUT), "w")
    for input_ids, token_type_ids, attention_mask, target in ynat_test_loader:
        input_ids, token_type_ids, attention_mask, target = (
            input_ids.to(device),
            token_type_ids.to(device),
            attention_mask.to(device),
            target.to(device),
        )
        output = model(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )[0]
        pred = output.argmax(dim=1)
        pred = pred.detach().cpu().numpy()

        for p in pred:
            output_file.write(f"{p}\n")
    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    # Container environment
    parser.add_argument(
        "--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/data")
    )
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument(
        "--model_tar_file",
        type=str,
        default="ynat.tar.gz",
        help="it needs to include all things for loading baseline model & tokenizer, \
             only supporting transformers.AutoModelForSequenceClassification as a model \
             transformers.XLMRobertaTokenizer or transformers.BertTokenizer as a tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/output"),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="maximum sequence length (default: 128)",
    )
    parser.add_argument(
        "--test_filename",
        default="ynat-v1.1_test.json",
        type=str,
        help="Name of the test file (default: ynat-v1.1_test.json)",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    inference(data_dir, model_dir, output_dir, args)
