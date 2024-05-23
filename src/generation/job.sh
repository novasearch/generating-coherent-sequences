#!/bin/bash

# Setup anaconda
eval "$(conda shell.bash hook)"

conda activate <env> # activate desired environment
cd <dir>/generation

python generate.py
