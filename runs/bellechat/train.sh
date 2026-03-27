#!/bin/bash
#
# Train bellechat end-to-end: tokenizer → pretrain → SFT → test.
# Run on an 8xH100 node after setup.sh and data upload.
#
# Usage:
#   bash runs/bellechat/train.sh                          # default d24
#   bash runs/bellechat/train.sh d20                      # smaller model
#   WANDB_RUN=my-run bash runs/bellechat/train.sh         # custom run name
#
set -e

export OMP_NUM_THREADS=1
export BELLECHAT_BASE_DIR="${BELLECHAT_BASE_DIR:-/workspace/bellechat/.cache/bellechat}"

DEPTH="${1:-24}"
WANDB_RUN="${WANDB_RUN:-bellechat-d${DEPTH}}"

cd /workspace/bellechat
source .venv/bin/activate

echo "=== bellechat training ==="
echo "Depth: d${DEPTH}"
echo "Wandb: ${WANDB_RUN}"
echo "Data:  ${BELLECHAT_BASE_DIR}"
echo ""

# Verify wandb
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. Training will proceed without logging."
    echo "         Set it with: export WANDB_API_KEY=your_key_here"
    echo ""
    WANDB_RUN=dummy
fi

# Verify data exists
SHARD_COUNT=$(ls "$BELLECHAT_BASE_DIR/shards/"*.parquet 2>/dev/null | wc -l | tr -d ' ')
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: No shards found in $BELLECHAT_BASE_DIR/shards/"
    echo "Upload data first. See runs/bellechat/setup.sh for instructions."
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

# Initialize report
python -m nanochat.report reset

# Step 1: Tokenizer
echo "=== [1/5] Training tokenizer ==="
python -m scripts.tok_train
python -m scripts.tok_eval

# Step 2: Pretrain
echo "=== [2/5] Pretraining (d${DEPTH}) ==="
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth="$DEPTH" \
    --device-batch-size=16 \
    --fp8 \
    --run="$WANDB_RUN"

# Step 3: Evaluate base model
echo "=== [3/5] Evaluating base model ==="
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# Step 4: SFT
echo "=== [4/5] Supervised fine-tuning ==="
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --chatcore-every=-1 \
    --run="$WANDB_RUN"

# Step 5: Test
echo "=== [5/5] Testing ==="
python -m scripts.chat_cli -p "Who are you?"
python -m scripts.chat_cli -p "Tell me about Mr. Charles Dickens"
python -m scripts.chat_cli -p "What is a computer?"
python -m scripts.chat_cli -p "Tell me about the Great War"

# Generate report
python -m nanochat.report generate

echo ""
echo "=== Training complete ==="
echo "Download the model from your local machine:"
echo "  rsync -avz --progress root@<POD_IP>:${BELLECHAT_BASE_DIR}/ ~/.cache/bellechat/ -e \"ssh -p <PORT> -i ~/.ssh/id_ed25519\" --exclude=\"shards/\""
