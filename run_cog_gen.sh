#!/bin/bash

# ---------------- 配置区域 ----------------
# [设置] 你想使用的固定 GPU ID 列表 (例如：0 1 2)
# 请确保这些 GPU 存在于你的系统并允许被占用。
FIXED_GPUS="1 2"

# 任务设置
ENV_NAME="cogvideo" 
PROJECT_DIR="/home/junjie/i2v"
# ----------------------------------------

# 1. 准备工作
mkdir -p "$PROJECT_DIR/logs"
# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME
# 切换到项目目录
cd $PROJECT_DIR

echo "====================================================="
echo "   Starting CogVideoX Generation Workers (Fixed GPUs)"
echo "====================================================="

# ---------------------------------------------------------
# 🤖 核心逻辑：使用固定的 GPU ID 列表
# ---------------------------------------------------------

# 将 FIXED_GPUS 字符串解析成数组
IFS=' ' read -r -a GPU_LIST <<< "$FIXED_GPUS"
NUM_GPUS=${#GPU_LIST[@]}

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "❌ FIXED_GPUS list is empty. Exiting."
    exit 1
fi

echo "✅ Using $NUM_GPUS fixed GPUs: ${GPU_LIST[*]}"
echo "====================================================="

# ---------------------------------------------------------
# 2. 循环启动进程
# ---------------------------------------------------------

for i in "${!GPU_LIST[@]}"; do
    gpu_id=${GPU_LIST[$i]}
    
    echo "🚀 Launching Worker $i on physical GPU $gpu_id"
    
    # 核心运行命令：利用 CUDA_VISIBLE_DEVICES 隔离 GPU。
    # Python 脚本在每个进程中只会看到并使用这一个 GPU (作为 cuda:0)。
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python i2v_cogx.py \
        > logs/worker_gpu${gpu_id}.log 2>&1 &
        
done

echo "====================================================="
echo "✅ All jobs launched! Monitor logs in $PROJECT_DIR/logs/"
echo "   To check status: jobs"
echo "====================================================="