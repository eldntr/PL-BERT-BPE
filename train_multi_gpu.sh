#!/bin/bash

# Script untuk menjalankan training dengan multi GPU menggunakan torchrun
# Usage: bash train_multi_gpu.sh
# Output disimpan ke file log di folder logs/

NUM_GPUS=1

# Buat folder logs jika belum ada
mkdir -p logs

# Generate timestamp untuk filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "🚀 Starting multi-GPU training with $NUM_GPUS GPUs..."
echo "Effective batch size: 64 x $NUM_GPUS = 64"
echo "Output akan disimpan ke: $LOG_FILE"
echo ""

# Jalankan training dan simpan output ke file (tee menampilkan + menyimpan ke file)
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py | tee "$LOG_FILE"

echo ""
echo "✓ Training finished!"
echo "Log file tersimpan di: $LOG_FILE"
