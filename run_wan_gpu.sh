#!/bin/bash
#SBATCH --job-name=wan_batch_gen
#SBATCH --output=logs/wan_batch_%j.out
#SBATCH --error=logs/wan_batch_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --mem=80G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

# Activate conda environment
source /users/hdu15/miniconda3/etc/profile.d/conda.sh
conda activate wan

# Print environment info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check GPU
nvidia-smi
echo ""

# Run the batch generation
cd /oscar/scratch/hdu15/i2v
python wan_batch_generate.py

echo ""
echo "Job completed!"
