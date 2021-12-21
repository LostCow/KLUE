from datasets import load_metric


def compute_metrics(pred):
    f1 = load_metric("f1")
    references = pred.label_ids
    predictions = pred.predictions.argmax(axis=1)
    metric = f1.compute(predictions=predictions, references=references, average="macro")
    return metric


def compute_metrics_for_smoothing(pred):
    f1 = load_metric("f1")
    references = pred.label_ids.argmax(axis=1)
    predictions = pred.predictions.argmax(axis=1)
    metric = f1.compute(predictions=predictions, references=references, average="macro")
    return metric
