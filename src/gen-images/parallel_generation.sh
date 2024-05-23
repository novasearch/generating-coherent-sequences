#!/bin/bash

NUM_WORKERS=$1

for (( WORKER_ID=0; WORKER_ID < NUM_WORKERS; WORKER_ID++ ))
do
    echo "Submitting job for worker $WORKER_ID out of $NUM_WORKERS"
    sbatch generate_images.sh $WORKER_ID $NUM_WORKERS
done