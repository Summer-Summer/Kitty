#!/bin/bash

source ./utils_channelkv.sh

echo "===================================="
echo "Evaluating MODEL: $MODEL"
echo "Running TASK: $TASK_NAME"
echo "GPUs: $GPUs"
echo "===================================="

mkdir -p logs

run_experiment    32 0  "Accuracy_PromoteRatios" 4   4   8

echo "All evaluations for $TASK_NAME completed."