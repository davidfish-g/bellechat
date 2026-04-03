#!/bin/bash
#
# One-time setup for a fresh RunPod instance.
#
# Usage:
#   git clone https://github.com/davidfish-g/bellechat.git /workspace/bellechat
#   cd /workspace/bellechat && bash runs/setup.sh
#
set -e

echo "=== bellechat setup ==="

cd /workspace/bellechat

# Install system deps
apt-get update -qq && apt-get install -y -qq rsync python3-dev > /dev/null 2>&1
echo "[1/5] System deps installed"

# Set up Python environment
pip install -q uv
uv sync --extra gpu --quiet
echo "[2/5] Python environment ready"

# Create data directories
mkdir -p .cache/bellechat/shards
echo "[3/5] Data directories created"

# Set environment
export BELLECHAT_BASE_DIR=/workspace/bellechat/.cache/bellechat
echo "export BELLECHAT_BASE_DIR=/workspace/bellechat/.cache/bellechat" >> ~/.bashrc
if [ -n "$WANDB_API_KEY" ]; then
    echo "export WANDB_API_KEY=$WANDB_API_KEY" >> ~/.bashrc
    echo "[4/5] Environment configured (wandb enabled)"
else
    echo "[4/5] WARNING: WANDB_API_KEY not set. Training logs won't be saved."
fi

# Verify GPUs
source .venv/bin/activate
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "[5/5] GPUs detected: $GPU_COUNT"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Now upload data from your local machine (in a separate terminal):"
echo ""
echo "  # Shards (~18 GB)"
echo "  rsync -avz --progress ~/.cache/bellechat/shards/ root@<POD_IP>:/workspace/bellechat/.cache/bellechat/shards/ -e \"ssh -p <PORT> -i ~/.ssh/id_ed25519\""
echo ""
echo "  # SFT conversations (~50 MB)"
echo "  rsync -avz ~/.cache/bellechat/*_conversations.jsonl root@<POD_IP>:/workspace/bellechat/.cache/bellechat/ -e \"ssh -p <PORT> -i ~/.ssh/id_ed25519\""
echo ""
echo "Then start training:"
echo "  export WANDB_API_KEY=your_key_here  # from https://wandb.ai/authorize"
echo "  bash runs/train.sh"
