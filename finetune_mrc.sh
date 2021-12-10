#!/bin/bash

python run_qa.py \
    --model_name_or_path klue/roberta-small \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --logging_strategy epoch \
    --save_strategy epoch \
    --dataset_name "klue" \
    --dataset_config_name 'mrc' \
    --fp16 \
    --save_total_limit 2 \
    --output_dir outputs \
    --version_2_with_negative \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --metric_for_best_model "eval_exact" \
    --load_best_model_at_end True