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
from dataloader import KlueStsDataLoaderFetcher
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig


def load_model_and_type(model_dir, model_tar_file):
    """load model and model type from tar file pre-fetched from s3

    Args:
        model_dir: str: the directory of tar file
        model_tar_path: str: the name of tar file
    """
    tarpath = os.path.join(model_dir, model_tar_file)
    tar = tarfile.open(tarpath, "r:gz")
    tar.extractall(path=model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
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
    klue_sts_dataloader_fetcher = KlueStsDataLoaderFetcher(tokenizer, args.max_length)
    kwargs = (
        {"num_workers": num_gpus, "pin_memory": True}
        if torch.cuda.is_available()
        else {}
    )
    klue_sts_test_loader = klue_sts_dataloader_fetcher.get_dataloader(
        file_path=os.path.join(data_dir, args.test_filename),
        batch_size=args.batch_size,
        **kwargs,
    )

    # infer
    output_file = open(os.path.join(output_dir, args.output_filename), "w")
    for out in klue_sts_test_loader:
        input_ids, attention_mask, token_type_ids, labels = [o.to(device) for o in out]
        if model_type == 'roberta':
            output = model(input_ids, attention_mask=attention_mask)[0]
        else:
            output = model(
                input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
            )[0]

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
    parser.add_argument(
        "--model_dir", type=str, default="./model"
    )
    parser.add_argument(
        "--model_tar_file",
        type=str,
        default="klue-sts.tar.gz",
        help="it needs to include all things for loading baseline model & tokenizer, \
             only supporting transformers.AutoModelForSequenceClassification as a model \
             transformers.XLMRobertaTokenizer or transformers.BertTokenizer as a tokenizer",
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/output")
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
