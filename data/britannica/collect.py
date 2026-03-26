"""
Download the 1911 Encyclopaedia Britannica from HuggingFace and split into articles.

The dataset (EdoVaira/Encyclopedia-Britannica) contains 72K entries from the 11th
edition as a single text file, one article per line. We split each line into an
individual cleaned text file.

Usage:
    python -m data.britannica.collect [--cache-dir DIR]
"""

import os
import re
import logging
import argparse
import unicodedata
import urllib.request

from data.common.cleaning import normalize_unicode, collapse_whitespace

logger = logging.getLogger(__name__)

DOWNLOAD_URL = "https://huggingface.co/datasets/EdoVaira/Encyclopedia-Britannica/resolve/main/combined_encyclopedia.txt"


def download_raw(cache_dir: str) -> str:
    """Download the raw combined encyclopedia file if not already cached."""
    raw_dir = os.path.join(cache_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, "combined_encyclopedia.txt")

    if os.path.exists(raw_path):
        logger.info(f"Already cached at {raw_path}")
        return raw_path

    logger.info(f"Downloading from {DOWNLOAD_URL}...")
    urllib.request.urlretrieve(DOWNLOAD_URL, raw_path)
    logger.info(f"Saved to {raw_path}")
    return raw_path


def split_and_clean(raw_path: str, output_dir: str, min_length: int = 50):
    """Split the combined file into individual article files."""
    os.makedirs(output_dir, exist_ok=True)

    with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    logger.info(f"Total lines: {len(lines):,}")

    saved = 0
    skipped = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if len(line) < min_length:
            skipped += 1
            continue

        text = normalize_unicode(line)
        text = collapse_whitespace(text)

        # Use first word(s) as filename
        words = text.split()
        title_words = []
        for w in words:
            if w.isupper() or w[0].isupper():
                title_words.append(w)
                if not w.isupper() or w.endswith(","):
                    break
            else:
                break
        title = " ".join(title_words).rstrip(",") or f"article_{i}"

        safe_name = re.sub(r'[<>:"/\\|?*]', "_", title)[:100]
        output_path = os.path.join(output_dir, f"{safe_name}.txt")

        # Handle duplicate filenames
        if os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{safe_name}_{i}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        saved += 1

    logger.info(f"Done: {saved:,} articles saved, {skipped:,} skipped (too short)")


def main():
    parser = argparse.ArgumentParser(description="Download and split 1911 Encyclopaedia Britannica")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/bellechat"),
                        help="Base cache directory")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <cache-dir>/britannica_clean)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output = args.output or os.path.join(args.cache_dir, "britannica", "clean")
    raw_path = download_raw(os.path.join(args.cache_dir, "britannica"))
    split_and_clean(raw_path, output)


if __name__ == "__main__":
    main()
