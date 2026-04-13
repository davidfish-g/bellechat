#!/bin/bash
#
# One-command setup and launch for bellechat.
# Run this from inside the cloned repo directory.
#
set -e

# Check Python is available
if ! command -v python3 &>/dev/null; then
    echo "Error: Python 3 is not installed. Download it from https://www.python.org/downloads/"
    exit 1
fi

# Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv and install dependencies
if [ ! -d ".venv" ]; then
    echo "Setting up Python environment..."
    uv sync --extra cpu --quiet
fi

# Download model weights if not already present
MODEL_DIR="$HOME/.cache/bellechat"
if [ ! -d "$MODEL_DIR/chatsft_checkpoints" ] || [ ! -d "$MODEL_DIR/tokenizer" ]; then
    echo "Downloading model from HuggingFace (~2 GB, this may take a few minutes)..."
    uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'david-fish/bellechat',
    local_dir='$MODEL_DIR',
    allow_patterns=['chatsft_checkpoints/**', 'tokenizer/**'],
)
"
    echo "Download complete."
fi

# Launch
echo ""
echo "Starting bellechat..."
echo "Open http://localhost:8000 in your browser to start chatting."
echo "Press Ctrl+C to stop."
echo ""
uv run python -m scripts.chat_web
