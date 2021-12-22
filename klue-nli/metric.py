from datasets import load_metric


def compute_metrics(pred):
    acc = load_metric("accuracy")
    references = pred.label_ids
    predictions = pred.predictions.argmax(axis=1)
    metric = acc.compute(predictions=predictions, references=references)
    return metric
