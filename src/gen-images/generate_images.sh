#!/bin/bash

# GPU Options gpu:3g.20gb:<1-12> or gpu:7g.40gb:<1-2> or gpu:nvidia_a100-pcie-40gb:1 # or shard:<nshards> or gpu:nvidia_a100-sxm4-40gb:<1-4> or gpu:nvidia_a100-pcie-40gb:<1-8>

PROMPTS_FILE='.json'
IMAGES_DIR=''
PROMPT_TYPE=''
WORKER_ID=$1
NUM_WORKERS=$2

eval "$(conda shell.bash hook)"
conda activate <env>
cd <dir>
python main.py $PROMPTS_FILE $IMAGES_DIR $PROMPT_TYPE $WORKER_ID $NUM_WORKERS
