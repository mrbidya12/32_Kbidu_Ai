#!/bin/bash
NUM_GPUS=${1:-1}
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 main.py