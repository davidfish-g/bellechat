#!/bin/bash
#
# Train bellechat end-to-end: tokenizer → pretrain → SFT → test.
# Run on an 8xH100 node after setup.sh and data upload.
#
# Usage:
#   bash runs/train.sh                          # default d24, no wandb
#   bash runs/train.sh d20                      # smaller model
#   WANDB_RUN=bellechat-v1 bash runs/train.sh   # with wandb logging
#
# Run inside screen for long training:
#   screen -S train bash runs/train.sh
#
set -e

export OMP_NUM_THREADS=1
export BELLECHAT_BASE_DIR="${BELLECHAT_BASE_DIR:-/workspace/bellechat/.cache/bellechat}"

DEPTH="${1:-24}"
WANDB_RUN="${WANDB_RUN:-dummy}"

cd /workspace/bellechat
source .venv/bin/activate

echo "=== bellechat training ==="
echo "Depth: d${DEPTH}"
echo "Wandb: ${WANDB_RUN}"
echo "Data:  ${BELLECHAT_BASE_DIR}"
echo ""

# Verify data exists
SHARD_COUNT=$(ls "$BELLECHAT_BASE_DIR/shards/"*.parquet 2>/dev/null | wc -l | tr -d ' ')
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: No shards found in $BELLECHAT_BASE_DIR/shards/"
    echo "Upload data first. See runs/setup.sh for instructions."
    exit 1
fi
echo "Found $SHARD_COUNT shards"

SFT_COUNT=0
for f in general_conversations.jsonl identity_conversations.jsonl boundary_conversations.jsonl; do
    if [ -f "$BELLECHAT_BASE_DIR/$f" ]; then
        SFT_COUNT=$((SFT_COUNT + 1))
    else
        echo "WARNING: Missing $f"
    fi
done
echo "Found $SFT_COUNT/3 SFT conversation files"
echo ""

# Step 1: Tokenizer
echo "=== [1/4] Training tokenizer ==="
python -m scripts.tok_train
python -m scripts.tok_eval

# Step 2: Pretrain
echo "=== [2/4] Pretraining (d${DEPTH}) ==="
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth="$DEPTH" \
    --device-batch-size=16 \
    --run="$WANDB_RUN"

# Step 3: SFT
echo "=== [3/4] Supervised fine-tuning ==="
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --chatcore-every=-1 \
    --run="$WANDB_RUN"

# Step 4: Test
echo "=== [4/4] Testing ==="
python -m scripts.chat_cli -p "Who are you?"
python -m scripts.chat_cli -p "Tell me about Mr. Charles Dickens"
python -m scripts.chat_cli -p "What is a computer?"
python -m scripts.chat_cli -p "Tell me about the Great War"

echo ""
echo "=== Training complete ==="
echo "Download the model from your local machine:"
echo "  rsync -avz --progress root@<POD_IP>:${BELLECHAT_BASE_DIR}/ ~/.cache/bellechat/ -e \"ssh -p <PORT> -i ~/.ssh/id_ed25519\" --include=\"*.pt\" --include=\"*/\" --exclude=\"*\""
