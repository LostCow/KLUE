from transformers import EvalPrediction
from utils_qa import postprocess_qa_predictions


class DataProcessor:
    def __init__(
        self,
        data_args,
        training_args,
        tokenizer,
        column_names,
    ):
        self.data_args = data_args
        self.training_args = training_args

        self.tokenizer = tokenizer
        self.question_column_name = "question" if "question" in column_names else column_names[0]
        self.context_column_name = "context" if "context" in column_names else column_names[1]
        self.answer_column_name = "answers" if "answers" in column_names else column_names[2]
        self.max_seq_length = min(self.data_args.max_seq_length, self.tokenizer.model_max_length)

        self.pad_on_right = self.tokenizer.padding_side == "right"

    def prepare_train_features_for_sketch_reader(self, examples):
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]

        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        tokenized_examples["labels"] = []

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        for i in range(len(tokenized_examples["input_ids"])):
            sample_index = sample_mapping[i]

            # answerable: 0, unanswerable: 1
            is_impossible = examples["is_impossible"][sample_index]
            tokenized_examples["labels"].append(0 if not is_impossible else 1)

        return tokenized_examples

    def prepare_eval_features_for_sketch_reader(self, examples):
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]

        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        tokenized_examples["labels"] = []
        tokenized_examples["example_id"] = []

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        for i in range(len(tokenized_examples["input_ids"])):
            sample_index = sample_mapping[i]

            id_col = examples["guid"][sample_index]
            tokenized_examples["example_id"].append(id_col)

            # answerable: 0, unanswerable: 1
            is_impossible = examples["is_impossible"][sample_index]
            tokenized_examples["labels"].append(0 if not is_impossible else 1)

        return tokenized_examples

    def prepare_test_features_for_sketch_reader(self, examples):
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]

        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        tokenized_examples["labels"] = []
        tokenized_examples["example_id"] = []

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        for i in range(len(tokenized_examples["input_ids"])):
            sample_index = sample_mapping[i]

            id_col = examples["guid"][sample_index]
            tokenized_examples["example_id"].append(id_col)

        return tokenized_examples

    def prepare_train_features_for_intensive_reader(self, examples):
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]

        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["is_impossible"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)

            sample_index = sample_mapping[i]
            answers = examples[self.answer_column_name][sample_index]
            is_impossible = examples["is_impossible"][sample_index]

            # is_impossible = "is_impossible"
            # tokenized_examples[is_impossible].append(examples[is_impossible][sample_index] - 1)  # 1부터 시작이므로

            # If no answers are given, set the cls_index as answer.
            if is_impossible or len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["is_impossible"].append(1.0)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["is_impossible"].append(1.0)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

                    tokenized_examples["is_impossible"].append(0.0)

        return tokenized_examples

    def prepare_eval_features_for_intensive_reader(self, examples):
        examples[self.question_column_name] = [q.lstrip() for q in examples[self.question_column_name]]

        tokenized_examples = self.tokenizer(
            examples[self.question_column_name if self.pad_on_right else self.context_column_name],
            examples[self.context_column_name if self.pad_on_right else self.question_column_name],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_seq_length,
            stride=self.data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if self.data_args.pad_to_max_length else False,
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["guid"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def post_processing_function(self, examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        log_level = self.training_args.get_process_log_level()

        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=self.data_args.version_2_with_negative,
            n_best_size=self.data_args.n_best_size,
            max_answer_length=self.data_args.max_answer_length,
            null_score_diff_threshold=self.data_args.null_score_diff_threshold,
            output_dir=self.training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if self.data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["guid"], "answers": ex[self.answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
