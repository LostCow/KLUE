"""Usage
$ python inference.py --data_dir data \
                      --model_dir model \
                      --output_dir output \
                      [args..]
"""
import argparse
import os
import tarfile

import torch
from torch.utils.data import DataLoader
from dataset import KlueStsWithSentenceMaskDataset
from transformers import AutoTokenizer, AutoConfig
from utils import read_json
from model import RobertaForStsRegression


def load_model_and_type(model_dir, model_tar_file):
    """load model and model type from tar file pre-fetched from s3

    Args:
        model_dir: str: the directory of tar file
        model_tar_path: str: the name of tar file
    """
    tarpath = os.path.join(model_dir, model_tar_file)
    tar = tarfile.open(tarpath, "r:gz")
    tar.extractall(path=model_dir)
    model = RobertaForStsRegression.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir)
    return model, config.model_type


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args) -> None:
    # configure gpu
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model, model_type = load_model_and_type(model_dir, args.model_tar_file)
    model.to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # get test_data_loader
    test_file_path = os.path.join(args.data_dir, args.test_filename)
    test_json = read_json(test_file_path)
    test_dataset = KlueStsWithSentenceMaskDataset(test_json, tokenizer, 510)
    data_loader = DataLoader(test_dataset, args.batch_size, drop_last=False)

    # infer
    output_file = open(os.path.join(output_dir, args.output_filename), "w")
    for batch in data_loader:
        input_data = {
            key: value.to(device) for key, value in batch.items() if not key == "labels"
        }

        output = model(**input_data)[0]

        preds = output.detach().cpu().numpy()

        for p in preds:
            score = p[0]
            output_file.write(f"{score}\n")

    output_file.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for inference (default: 64)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/data")
    )
    parser.add_argument("--model_dir", type=str, default="./model")
    parser.add_argument(
        "--model_tar_file",
        type=str,
        default="klue-sts.tar.gz",
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
        default=510,
        help="maximum sequence length (default: 510)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="output.csv",
        help="filename of the inference output (default: output.csv)",
    )
    parser.add_argument(
        "--test_filename",
        default="klue-sts-v1.1_test.json",
        type=str,
        help="Name of the test file (default: klue-sts-v1.1_test.json)",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    inference(data_dir, model_dir, output_dir, args)


if __name__ == "__main__":
    main()
