from datasets.load import load_metric
from transformers.trainer_utils import EvalPrediction


def compute_metrics(p: EvalPrediction):
    metric = load_metric("squad_v2")
    return metric.compute(predictions=p.predictions, references=p.label_ids)
