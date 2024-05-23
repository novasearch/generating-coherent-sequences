#!/bin/bash

eval "$(conda shell.bash hook)"

conda activate <env>

cd <dir>

python3 instruct_blip_captioning_windowed.py