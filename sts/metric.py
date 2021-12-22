from datasets import load_metric


def compute_metrics(pred):
    pearson = load_metric("pearsonr").compute
    references = pred.label_ids
    predictions = pred.predictions
    metric = pearson(predictions=predictions, references=references)
    return metric
