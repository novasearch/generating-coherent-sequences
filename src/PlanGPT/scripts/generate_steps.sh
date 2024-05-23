#!/bin/bash

# Setup anaconda
eval "$(conda shell.bash hook)"

conda activate <env> # activate desired environment
cd <dir>/PlanGPT # change dir to where we want to run scripts

RUN_NAME=
CONFIG=greedy
PEFT_MODEL_ID=<model_id>

echo $CONFIG
python src/inference/inference_window.py $PEFT_MODEL_ID $RUN_NAME $CONFIG