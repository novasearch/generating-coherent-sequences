#!/bin/bash

# Setup anaconda
eval "$(conda shell.bash hook)"

conda activate <env> # activate desired environment
cd <dir>/PlanGPT # change dir to where we want to run scripts

BASE_MODEL=""

CUDA_LAUNCH_BLOCKING=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port 55697 src/main.py \
    --run_name "" \
    --project_name "" \
    --base_model $BASE_MODEL \
    --output_dir "." \
    --overwrite_output_dir True \
    --data_path <data> \
    --dataset_name <dataset_name> \
    --dataset_type <dataset_type> \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --eval_steps 500 \
    --save_strategy "epoch" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --model_max_length 400 \
    --warmup_steps 0 \
    --report_to_wandb True \
    --infer_checkpoints False \
    --mixed_precision BF16 \
    --warmup_before_inference 0 \
    --reload_optimizer True \
    --use_dpo False \
    --debug False \
    --parallel_type NO \
    --lora True
