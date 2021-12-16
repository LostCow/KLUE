from typing import Dict, List, Optional, Union, Any, Tuple

from torch import nn
import numpy as np
from torch.nn.modules import module

from transformers import (
    Trainer,
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    PredictionOutput,
    EvalPrediction,
    speed_metrics,
    denumpify_detensorize,
)
import datasets
from datasets.load import load_metric
from datasets import Dataset

from processor import DataProcessor
from model import (
    RobertaForSequenceClassification,
    RobertaForQuestionAnsweringAVPool,
    ElectraForQuestionAnsweringAVPool,
)

import collections
import time
from tqdm import tqdm
import os
import json

from copy import copy


class SketchReader(Trainer):
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples

    def post_process_function(
        self,
        output: Union[np.ndarray, EvalLoopOutput],
        eval_examples: Dataset,
        eval_dataset: Dataset,
        mode="eval",
    ):
        if isinstance(output, EvalLoopOutput):
            logits = output.predictions
        else:
            logits = output
        example_id_to_index = {k: i for i, k in enumerate(eval_examples["guid"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(eval_dataset):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        count_map = {k: len(v) for k, v in features_per_example.items()}

        logits_ans = np.zeros(len(count_map))
        logits_na = np.zeros(len(count_map))
        for example_index, example in enumerate(tqdm(eval_examples)):
            feature_indices = features_per_example[example_index]
            n_strides = count_map[example_index]
            logits_ans[example_index] += logits[example_index, 0] / n_strides
            logits_na[example_index] += logits[example_index, 1] / n_strides

        # Calculate E-FV score
        score_ext = logits_ans - logits_na

        # Save external front verification score
        final_map = dict(zip(eval_examples["guid"], score_ext.tolist()))
        with open(os.path.join(self.args.output_dir, "cls_score.json"), "w") as writer:
            writer.write(json.dumps(final_map, indent=4) + "\n")
        if mode == "evaluate":
            return EvalPrediction(
                predictions=logits,
                label_ids=output.label_ids,
            )
        else:
            return final_map

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            metrics = self.compute_metrics(output)

            eval_preds = self.post_process_function(
                eval_examples=eval_examples, eval_dataset=eval_dataset, output=output
            )

            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        else:
            metrics = {}

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics


class IntensiveReader(Trainer):
    def __init__(self, *args, eval_examples=None, data_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.data_args = data_args

    def post_process_function(
        self,
        output: Union[np.ndarray, EvalLoopOutput],
        eval_examples: Dataset,
        eval_dataset: Dataset,
        mode="eval",
    ) -> Union[List[Dict[str, Any]], EvalPrediction]:

        predictions, _, _, scores_diff_json = self.compute_predictions(
            eval_examples,
            eval_dataset,
            output.predictions,
            version_2_with_negative=self.data_args.version_2_with_negative,
            n_best_size=self.data_args.n_best_size,
            max_answer_length=self.data_args.max_answer_length,
            null_score_diff_threshold=self.data_args.null_score_diff_threshold,
            output_dir=self.args.output_dir,
        )

        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": scores_diff_json[k]}
            for k, v in predictions.items()
        ]
        if mode == "predict":
            return formatted_predictions
        else:
            references = [{"id": ex["guid"], "answers": ex["answers"]} for ex in eval_examples]
            return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    def compute_predictions(
        self,
        examples: Dataset,
        features: Dataset,
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        use_choice_logits: bool = False,
    ):
        # Threshold-based Answerable Verification (TAV)
        if len(predictions) not in [2, 3]:
            raise ValueError(
                "`predictions` should be a tuple with two or three elements "
                "(start_logits, end_logits, choice_logits)."
            )
        all_start_logits, all_end_logits = predictions[:2]
        all_choice_logits = None
        if len(predictions) == 3:
            all_choice_logits = predictions[-1]

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["guid"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict() if version_2_with_negative else None

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_prediction = None
            prelim_predictions = []

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # score_null = s1 + e1
                feature_null_score = start_logits[0] + end_logits[0]
                if all_choice_logits is not None:
                    choice_logits = all_choice_logits[feature_index]
                if use_choice_logits:
                    feature_null_score = choice_logits[1]
                # This is what will allow us to map some the positions
                # in our logits to span of texts in the original context.
                offset_mapping = features[feature_index]["offset_mapping"]
                # Optional `token_is_max_context`,
                # if provided we will remove answers that do not have the maximum context
                # available in the current feature.
                token_is_max_context = features[feature_index].get("token_is_max_context", None)

                # Update minimum null prediction.
                if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                # Go through all possibilities for the {top k} greater start and end logits
                # top k = n_best_size if not beam_based else n_start_top, n_end_top
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers!
                        # either because the indices are out of bounds
                        # or correspond to part of the input_ids that are note in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length negative or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue
                        # Don't consider answer that don't have the maximum context available
                        # (if such information is provided).
                        if token_is_max_context is not None and not token_is_max_context.get(
                            str(start_index), False
                        ):
                            continue
                        prelim_predictions.append(
                            {
                                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )

            if version_2_with_negative:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # Only keep the best `n_best_size` predictions
            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score.
            if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
                predictions.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]

            # In the very rare edge case we have not a single non-null prediction,
            # we create a fake prediction to avoid failure.
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                predictions.insert(
                    0,
                    {
                        "text": "",
                        "start_logit": 0.0,
                        "end_logit": 0.0,
                        "score": 0.0,
                    },
                )

            # Compute the softmax of all scores
            # (we do it with numpy to stay independent from torch/tf) in this file,
            #  using the LogSum trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not version_2_with_negative:
                all_predictions[example["guid"]] = predictions[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction.
                # print(i, len(predictions), type(predictions), predictions, predictions[0])
                i = 0
                while i < len(predictions) and predictions[i]["text"] == "":  # i == 2, len(predictions)=2
                    i += 1

                if i != len(predictions):
                    best_non_null_pred = predictions[i]

                    # Then we compare to the null prediction using the threshold.
                    score_diff = (
                        null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                    )
                    scores_diff_json[example["guid"]] = float(score_diff)  # To be JSON-serializable.
                    if score_diff > null_score_diff_threshold:
                        all_predictions[example["guid"]] = ""
                    else:
                        all_predictions[example["guid"]] = best_non_null_pred["text"]
                else:
                    scores_diff_json[example["guid"]] = float(null_score)
                    all_predictions[example["guid"]] = ""

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example["guid"]] = [
                {
                    k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v)
                    for k, v in pred.items()
                }
                for pred in predictions
            ]

        # If we have an output_dir, let's save all those dicts.
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise EnvironmentError(f"{output_dir} is not a directory.")

            prediction_file = os.path.join(output_dir, "predictions.json")
            nbest_file = os.path.join(output_dir, "nbest_predictions.json")
            if version_2_with_negative:
                null_odds_file = os.path.join(output_dir, "null_odds.json")

            with open(prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")
            with open(nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
            if version_2_with_negative:
                with open(null_odds_file, "w") as writer:
                    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

        return all_predictions, all_nbest_json, scores_diff_json, scores_diff_json

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(eval_dataset, Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        eval_preds = self.post_process_function(output, eval_examples, eval_dataset)

        metrics = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(eval_preds)
            # To be JSON-serializable, we need to remove numpy types or zero-d tensors
            metrics = denumpify_detensorize(metrics)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)

        # Log and save evaluation results
        filename = "eval_results.txt"
        eval_result_file = "intensive_reader_" + filename
        with open(os.path.join(self.args.output_dir, eval_result_file), "a") as writer:
            for key in sorted(metrics.keys()):
                writer.write("%s = %s\n" % (key, str(metrics[key])))
            writer.write("\n")

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics


# class RearVerifier:
#     def __init__(
#         self,
#         beta1: int = 1,
#         beta2: int = 1,
#         best_cof: int = 1,
#     ):
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.best_cof = best_cof

#     def __call__(
#         self,
#         score_ext: Dict[str, float],
#         score_diff: Dict[str, float],
#         nbest_preds: Dict[str, Dict[int, Dict[str, float]]],
#     ):
#         all_scores = collections.OrderedDict()
#         assert score_ext.keys() == score_diff.keys()
#         for key in score_ext.keys():
#             if key not in all_scores:
#                 all_scores[key] = []
#             all_scores[key].append([self.beta1 * score_ext[key], self.beta2 * score_diff[key]])
#         output_scores = {}
#         for key, scores in all_scores.items():
#             mean_score = sum(scores) / float(len(scores))
#             output_scores[key] = mean_score

#         all_nbest = collections.OrderedDict()
#         for key, entries in nbest_preds.items():
#             if key not in all_nbest:
#                 all_nbest[key] = collections.defaultdict(float)
#             for entry in entries:
#                 prob = self.best_cof * entry["probability"]
#                 all_nbest[key][entry["text"]] += prob

#         output_predictions = {}
#         for key, entry_map in all_nbest.items():
#             sorted_texts = sorted(entry_map.keys(), key=lambda x: entry_map[x], reverse=True)
#             best_text = sorted_texts[0]
#             output_predictions[key] = best_text

#         for qid in output_predictions.keys():
#             # if output_scores[qid] > thresh:
#             if output_scores[qid] > 1:
#                 output_predictions[qid] = ""

#         return output_predictions, output_scores


class RetroReader:
    def __init__(
        self,
        sketch_model_name_or_path="klue/roberta-large",
        intensive_model_name_or_path="klue/roberta-large",
        base_training_args=None,
        data_args=None,
        train_examples=None,
        eval_examples=None,
        test_examples=None,
    ):
        self.sketch_model_name_or_path = sketch_model_name_or_path
        self.intensive_model_name_or_path = intensive_model_name_or_path
        self.base_training_args = base_training_args
        self.data_args = data_args
        self.train_examples = train_examples
        self.eval_examples = eval_examples
        self.test_examples = test_examples

        self.column_names = self.train_examples.column_names

        self.sketch_tokenizer = AutoTokenizer.from_pretrained(sketch_model_name_or_path, use_fast=True)
        self.intensive_tokenizer = AutoTokenizer.from_pretrained(intensive_model_name_or_path, use_fast=True)

        self.sketch_data_collator = (
            default_data_collator
            if data_args.pad_to_max_length
            else DataCollatorWithPadding(
                self.sketch_tokenizer, pad_to_multiple_of=8 if self.base_training_args.fp16 else None
            )
        )
        self.intensive_data_collator = (
            default_data_collator
            if data_args.pad_to_max_length
            else DataCollatorWithPadding(
                self.intensive_tokenizer, pad_to_multiple_of=8 if self.base_training_args.fp16 else None
            )
        )

        self.mrc_processor = DataProcessor(
            data_args=self.data_args,
            sketch_tokenizer=self.sketch_tokenizer,
            intensive_tokenizer=self.intensive_tokenizer,
            column_names=self.column_names,
        )

        self.init_module("sketch")
        self.init_module("intensive")

    def train(self):
        self.sketch_reader.train()
        self.intensive_reader.train()

    def preprocess_examples(self, module_name="sketch"):
        with self.base_training_args.main_process_first(
            desc=f"train dataset for {module_name} reader map pre-processing"
        ):
            train_dataset = self.train_examples.map(
                self.mrc_processor.prepare_train_features_for_sketch_reader
                if module_name == "sketch"
                else self.mrc_processor.prepare_train_features_for_intensive_reader,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc=f"Running tokenizer on train dataset for {module_name} reader",
            )

        with self.base_training_args.main_process_first(
            desc=f"validation dataset for {module_name} reader map pre-processing"
        ):
            eval_dataset = self.eval_examples.map(
                self.mrc_processor.prepare_eval_features_for_sketch_reader
                if module_name == "sketch"
                else self.mrc_processor.prepare_eval_features_for_intensive_reader,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc=f"Running tokenizer on validation dataset for {module_name} reader",
            )
        return train_dataset, eval_dataset

    def init_module(self, module_name="sketch"):
        if module_name == "sketch":
            sketch_reader_config = AutoConfig.from_pretrained(self.sketch_model_name_or_path, num_labels=2)

            if "electra" in self.sketch_model_name_or_path:
                sketch_reader_model = AutoModelForSequenceClassification.from_pretrained(
                    self.sketch_model_name_or_path, config=sketch_reader_config
                )
            else:
                sketch_reader_model = RobertaForSequenceClassification.from_pretrained(
                    self.sketch_model_name_or_path, config=sketch_reader_config
                )
            (
                self.train_dataset_for_sketch_reader,
                self.eval_dataset_for_sketch_reader,
            ) = self.preprocess_examples(module_name="sketch")

            accuracy = datasets.load_metric("accuracy").compute
            precision = datasets.load_metric("precision").compute
            recall = datasets.load_metric("recall").compute
            f1 = datasets.load_metric("f1").compute

            def compute_metrics_for_sketch_reader(p: EvalPrediction):

                predictions = p.predictions.argmax(axis=1)
                references = p.label_ids
                metric = accuracy(predictions=predictions, references=references)
                metric.update(precision(predictions=predictions, references=references))
                metric.update(recall(predictions=predictions, references=references))
                metric.update(f1(predictions=predictions, references=references))

                return metric

            sketch_reader_args = copy(self.base_training_args)
            sketch_reader_args.metric_for_best_model = "eval_f1"
            sketch_reader_args.output_dir = "sketch_reader_outputs"
            self.sketch_reader = SketchReader(
                model=sketch_reader_model,
                args=sketch_reader_args,
                train_dataset=self.train_dataset_for_sketch_reader if sketch_reader_args.do_train else None,
                eval_dataset=self.eval_dataset_for_sketch_reader if sketch_reader_args.do_eval else None,
                eval_examples=self.eval_examples if sketch_reader_args.do_eval else None,
                tokenizer=self.sketch_tokenizer,
                data_collator=self.sketch_data_collator,
                compute_metrics=compute_metrics_for_sketch_reader,
            )

        elif module_name == "intensive":
            intensive_reader_config = AutoConfig.from_pretrained(self.intensive_model_name_or_path)
            if "electra" in self.intensive_model_name_or_path:
                intensive_reader_model = ElectraForQuestionAnsweringAVPool.from_pretrained(
                    self.intensive_model_name_or_path, config=intensive_reader_config
                )
            else:
                intensive_reader_model = RobertaForQuestionAnsweringAVPool.from_pretrained(
                    self.intensive_model_name_or_path, config=intensive_reader_config
                )
            (
                self.train_dataset_for_intensive_reader,
                self.eval_dataset_for_intensive_reader,
            ) = self.preprocess_examples(module_name="intensive")

            def compute_metrics(p: EvalPrediction):
                metric = load_metric("squad_v2")
                return metric.compute(predictions=p.predictions, references=p.label_ids)

            intensive_reader_args = copy(self.base_training_args)
            intensive_reader_args.metric_for_best_model = "eval_exact"
            intensive_reader_args.output_dir = "intensive_reader_outputs"
            self.intensive_reader = IntensiveReader(
                model=intensive_reader_model,
                args=intensive_reader_args,
                data_args=self.data_args,
                train_dataset=self.train_dataset_for_intensive_reader
                if intensive_reader_args.do_train
                else None,
                eval_dataset=self.eval_dataset_for_intensive_reader
                if intensive_reader_args.do_eval
                else None,
                eval_examples=self.eval_examples if intensive_reader_args.do_eval else None,
                tokenizer=self.intensive_tokenizer,
                data_collator=self.intensive_data_collator,
                compute_metrics=compute_metrics,
            )
