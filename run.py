"""One-command setup and launch for bellechat."""

import shutil
import subprocess
import sys
import os
import threading
import time
import urllib.request
import webbrowser

def main():
    # Install uv if not present; pip may put it somewhere off PATH, so run it via `python -m uv`
    if shutil.which("uv"):
        uv = ["uv"]
    else:
        print("Installing uv (Python package manager)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
        uv = [sys.executable, "-m", "uv"]

    # Create venv and install dependencies
    if not os.path.isdir(".venv"):
        print("Setting up Python environment...")
        subprocess.run([*uv, "sync", "--extra", "cpu", "--quiet"], check=True)

    # Download model weights if not already present
    model_dir = os.path.join(os.path.expanduser("~"), ".cache", "bellechat")
    has_checkpoint = os.path.isdir(os.path.join(model_dir, "chatsft_checkpoints"))
    has_tokenizer = os.path.isdir(os.path.join(model_dir, "tokenizer"))
    if not has_checkpoint or not has_tokenizer:
        print("Downloading model from HuggingFace (~4 GB, this may take a few minutes)...")
        # Only the weights and metadata; the optimizer states (~6 GB) are just for resuming training
        subprocess.run([*uv, "run", "python", "-c", f"""
from huggingface_hub import snapshot_download
snapshot_download(
    'david-fish/bellechat',
    local_dir={model_dir!r},
    allow_patterns=['chatsft_checkpoints/*/model_*.pt', 'chatsft_checkpoints/*/meta_*.json', 'tokenizer/*'],
)
"""], check=True)
        print("Download complete.")

    # Open browser once server is ready (it loads the model first, which can take a few minutes on CPU)
    def open_browser():
        for _ in range(300):
            try:
                urllib.request.urlopen("http://localhost:8000/health")
                webbrowser.open("http://localhost:8000")
                return
            except Exception:
                time.sleep(1)
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch
    print()
    print("Starting bellechat...")
    print("Press Ctrl+C to stop.")
    print()
    try:
        subprocess.run([*uv, "run", "python", "-m", "scripts.chat_web"], check=True)
    except KeyboardInterrupt:
        pass  # Ctrl+C already stopped the server cleanly

if __name__ == "__main__":
    main()
