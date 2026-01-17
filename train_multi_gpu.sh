#!/bin/bash

# Script untuk menjalankan training dengan 4 GPU menggunakan torchrun
# Usage: bash train_multi_gpu.sh

NUM_GPUS=1

echo "🚀 Starting multi-GPU training with $NUM_GPUS GPUs..."
echo "Effective batch size: 192 x $NUM_GPUS = 768"
echo ""

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py

echo ""
echo "✓ Training finished!"
