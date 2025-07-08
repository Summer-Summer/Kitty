#!/bin/bash

source ./utils_channelkv.sh

echo "===================================="
echo "Evaluating MODEL: $MODEL"
echo "Running TASK: $TASK_NAME"
echo "GPUs: $GPUs"
echo "===================================="

mkdir -p logs

#run_baseline    32 0  "Sensitivity Check" 2   2
#run_baseline    32 0  "Sensitivity Check" 4   2
#run_baseline    32 0  "Sensitivity Check" 6   2
#run_baseline    32 0  "Sensitivity Check" 8   2
#run_baseline    32 0  "Sensitivity Check" 16  2
#
#run_baseline    32 0  "Sensitivity Check" 2   3
#run_baseline    32 0  "Sensitivity Check" 4   3
#run_baseline    32 0  "Sensitivity Check" 6   3
#run_baseline    32 0  "Sensitivity Check" 8   3
#run_baseline    32 0  "Sensitivity Check" 16  3

#run_baseline    32 0  "Sensitivity Check" 2   4
#run_baseline    32 0  "Sensitivity Check" 4   4
#run_baseline    32 0  "Sensitivity Check" 6   4
#run_baseline    32 0  "Sensitivity Check" 8   4
#run_baseline    32 0  "Sensitivity Check" 16  4

#run_baseline    32 0  "Sensitivity Check" 2   6
#run_baseline    32 0  "Sensitivity Check" 4   6
#run_baseline    32 0  "Sensitivity Check" 6   6
#run_baseline    32 0  "Sensitivity Check" 8   6
#run_baseline    32 0  "Sensitivity Check" 16  6

run_baseline    32 0  "Sensitivity Check" 2   8
run_baseline    32 0  "Sensitivity Check" 4   8
run_baseline    32 0  "Sensitivity Check" 6   8
run_baseline    32 0  "Sensitivity Check" 8   8
run_baseline    32 0  "Sensitivity Check" 16  8

#run_baseline    32 0  "Sensitivity Check" 2   16
#run_baseline    32 0  "Sensitivity Check" 4   16
#run_baseline    32 0  "Sensitivity Check" 6   16
#run_baseline    32 0  "Sensitivity Check" 8   16
#run_baseline    32 0  "Sensitivity Check" 16  16

echo "All evaluations for $TASK_NAME completed."