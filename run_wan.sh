#!/bin/bash

# Activate wan conda environment and run batch generation
source /users/hdu15/miniconda3/etc/profile.d/conda.sh
conda activate wan

# Check if environment activated successfully
if [ $? -ne 0 ]; then
    echo "Failed to activate wan environment"
    exit 1
fi

echo "Environment: wan"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run the batch generation script
python wan_batch_generate.py "$@"
