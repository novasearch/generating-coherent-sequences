#!/bin/bash

# Setup anaconda
eval "$(conda shell.bash hook)"

conda activate bordalo-thesis # activate desired environment
cd <dir> # change dir to where we want to run scripts

MODEL_ID=<checkpoint>
EVAL_DATASET=<eval_dataset>.json
CONFIG=greedy
RESULT_FILE=results/annotations/<result_file>

echo "Saving to $RESULT_FILE.json"
python src/inference/inference_eval_special.py $MODEL_ID $EVAL_DATASET $RESULT_FILE $CONFIG
