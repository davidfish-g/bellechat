"""One-command setup and launch for bellechat."""

import subprocess
import sys
import os
import threading
import time
import urllib.request
import webbrowser

def main():
    # Install uv if not present
    if not any(os.access(os.path.join(d, "uv"), os.X_OK) for d in os.environ.get("PATH", "").split(os.pathsep)):
        print("Installing uv (Python package manager)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)

    # Create venv and install dependencies
    if not os.path.isdir(".venv"):
        print("Setting up Python environment...")
        subprocess.run(["uv", "sync", "--extra", "cpu", "--quiet"], check=True)

    # Download model weights if not already present
    model_dir = os.path.join(os.path.expanduser("~"), ".cache", "bellechat")
    has_checkpoint = os.path.isdir(os.path.join(model_dir, "chatsft_checkpoints"))
    has_tokenizer = os.path.isdir(os.path.join(model_dir, "tokenizer"))
    if not has_checkpoint or not has_tokenizer:
        print("Downloading model from HuggingFace (~2 GB, this may take a few minutes)...")
        subprocess.run(["uv", "run", "python", "-c", f"""
from huggingface_hub import snapshot_download
snapshot_download(
    'david-fish/bellechat',
    local_dir='{model_dir}',
    allow_patterns=['chatsft_checkpoints/**', 'tokenizer/**'],
)
"""], check=True)
        print("Download complete.")

    # Open browser once server is ready
    def open_browser():
        for _ in range(30):
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
    subprocess.run(["uv", "run", "python", "-m", "scripts.chat_web"], check=True)

if __name__ == "__main__":
    main()
