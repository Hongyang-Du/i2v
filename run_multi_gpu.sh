#!/bin/bash

# Multi-GPU parallel execution script for wan_batch_generate.py
# Usage: bash run_multi_gpu.sh [num_gpus]

NUM_GPUS=${1:-$(nvidia-smi --list-gpus | wc -l)}

echo "Starting parallel generation on $NUM_GPUS GPUs"
echo "=============================================="

# Read total number of samples from JSON
TOTAL_SAMPLES=$(python3 -c "import json; data=json.load(open('generated_prompts.json')); print(len(data))")
echo "Total samples: $TOTAL_SAMPLES"

# Calculate samples per GPU
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / NUM_GPUS))
REMAINDER=$((TOTAL_SAMPLES % NUM_GPUS))

echo "Samples per GPU: ~$SAMPLES_PER_GPU"
echo "=============================================="

# Launch processes on each GPU
for ((gpu_id=0; gpu_id<NUM_GPUS; gpu_id++)); do
    # Calculate start and end index for this GPU
    START_IDX=$((gpu_id * SAMPLES_PER_GPU))

    if [ $gpu_id -eq $((NUM_GPUS - 1)) ]; then
        # Last GPU handles remainder
        END_IDX=$TOTAL_SAMPLES
    else
        END_IDX=$(((gpu_id + 1) * SAMPLES_PER_GPU))
    fi

    NUM_SAMPLES=$((END_IDX - START_IDX))

    echo "GPU $gpu_id: Processing samples $START_IDX to $((END_IDX - 1)) ($NUM_SAMPLES samples)"

    # Launch process in background with specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python3 wan_batch_generate.py \
        --start_idx $START_IDX \
        --end_idx $END_IDX \
        --gpu_id $gpu_id \
        > logs/gpu_${gpu_id}.log 2>&1 &

    PID=$!
    echo "  Launched process $PID on GPU $gpu_id"
done

echo "=============================================="
echo "All processes launched. Monitor with:"
echo "  tail -f logs/gpu_*.log"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Wait for all processes to complete..."
wait

echo "=============================================="
echo "All GPU processes completed!"
echo "Results saved in wan_generation_results.json"
