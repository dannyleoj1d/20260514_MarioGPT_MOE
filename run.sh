#!/bin/bash
# =====================================================
#  MoE MarioGPT - Cloud Startup Script
#  Compatible with: RunPod / vast.ai / Lambda Labs
#  Recommended GPU: A100 40GB or above
# =====================================================
set -e

WORKDIR="/workspace/mario-gpt-moe-mlp-2DROPE"
OUTPUT_DIR="/workspace/output"

echo "=== [1/4] Installing dependencies ==="
pip install --upgrade pip -q
pip install -r "$WORKDIR/requirements.txt" -q

echo "=== [2/4] Installing package ==="
cd "$WORKDIR"
pip install -e . -q

echo "=== [3/4] Checking GPU ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "=== [4/4] Starting training ==="
mkdir -p "$OUTPUT_DIR"

python train_cloud.py \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 16 \
    --total_steps 100000 \
    --save_iter 10000 \
    --num_experts 8 \
    --top_k 2 \
    --mixed_precision bf16 \
    --lr 5e-4
