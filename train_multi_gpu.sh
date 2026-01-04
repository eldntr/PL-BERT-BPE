#!/bin/bash

# Script untuk menjalankan training dengan 4 GPU menggunakan torchrun
# Usage: bash train_multi_gpu.sh

NUM_GPUS=1

# Disable tokenizers parallelism warning saat fork
export TOKENIZERS_PARALLELISM=false

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py
