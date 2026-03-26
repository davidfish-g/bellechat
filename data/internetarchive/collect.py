"""
Collect pre-1914 English texts from the Internet Archive.

Uses the `internetarchive` Python library to search for texts published
before 1914, download the DjVu OCR text, clean it, and save to output directory.

Usage:
    python -m data.internetarchive.collect [--max-items 50000] [--workers 8]
"""

import os
import re
import time
import logging
import argparse
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

import internetarchive as ia

from data.common.cleaning import (
    normalize_unicode,
    collapse_whitespace,
    is_quality_text,
)

logger = logging.getLogger(__name__)


def search_pre1914_texts(max_items: int = 10000) -> list[str]:
    """
    Search Internet Archive for pre-1914 English texts.
    Returns a list of item identifiers.
    """
    query = "date:[1800-01-01 TO 1913-12-31] language:eng mediatype:texts"
    logger.info(f"Searching IA: {query}")

    identifiers = []
    results = ia.search_items(query, fields=["identifier"])

    for item in results:
        identifiers.append(item["identifier"])
        if len(identifiers) >= max_items:
            break
        if len(identifiers) % 10000 == 0:
            logger.info(f"  ... {len(identifiers)} identifiers collected")

    logger.info(f"Found {len(identifiers)} items (limited to {max_items})")
    return identifiers


def download_and_clean(identifier: str, output_dir: str) -> tuple[bool, str]:
    """
    Download the DjVu text for an IA item, clean it, and save.
    Returns (success, message).
    """
    import urllib.request

    output_path = os.path.join(output_dir, f"{identifier}.txt")
    if os.path.exists(output_path):
        return True, "already exists"

    try:
        item = ia.get_item(identifier)
    except Exception as e:
        return False, f"metadata error: {e}"

    # Find the _djvu.txt file
    djvu_file = None
    for f in item.files:
        if f["name"].endswith("_djvu.txt"):
            djvu_file = f
            break

    if djvu_file is None:
        return False, "no djvu.txt"

    # Skip very small or very large files
    size = int(djvu_file.get("size", 0))
    if size < 1000:
        return False, f"too small ({size} bytes)"
    if size > 50_000_000:  # 50MB
        return False, f"too large ({size} bytes)"

    # Download the text
    try:
        url = f"https://archive.org/download/{identifier}/{djvu_file['name']}"
        req = urllib.request.Request(url, headers={"User-Agent": "bellechat/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
    except Exception as e:
        return False, f"download error: {e}"

    # Clean
    text = normalize_unicode(text)
    text = collapse_whitespace(text)

    # Quality check
    if not is_quality_text(text, min_length=1000):
        return False, f"failed quality check (len={len(text)})"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return True, f"ok ({len(text):,} chars)"


def _load_processed_log(log_path: str) -> set[str]:
    """Load the set of already-processed identifiers."""
    if not os.path.exists(log_path):
        return set()
    with open(log_path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def _append_to_log(log_path: str, identifier: str):
    """Append a processed identifier to the log (thread-safe via append mode)."""
    with open(log_path, "a") as f:
        f.write(identifier + "\n")


def collect(output_dir: str, max_items: int = 10000, max_workers: int = 4):
    """Search and download pre-1914 texts from Internet Archive."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "processed_ids.txt")

    already_processed = _load_processed_log(log_path)
    logger.info(f"Already processed: {len(already_processed)} identifiers")

    identifiers = search_pre1914_texts(max_items=max_items)

    # Filter out already-processed identifiers
    identifiers = [i for i in identifiers if i not in already_processed]
    logger.info(f"New items to process: {len(identifiers)} (after filtering)")
    logger.info(f"Downloading with {max_workers} workers")
    logger.info(f"Output: {output_dir}")

    if not identifiers:
        logger.info("Nothing new to download.")
        return

    success = 0
    skip = 0
    fail = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(download_and_clean, ident, output_dir): ident
            for ident in identifiers
        }

        for future in as_completed(futures):
            ident = futures[future]
            ok, msg = future.result()
            if "already exists" in msg:
                skip += 1
            elif ok:
                success += 1
            else:
                fail += 1

            # Log every identifier we've attempted (success or fail)
            _append_to_log(log_path, ident)

            total_done = success + fail + skip
            if total_done % 500 == 0:
                elapsed = time.time() - t0
                rate = total_done / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {total_done}/{len(identifiers)} "
                    f"({success} ok, {fail} fail, {skip} skip) "
                    f"[{rate:.1f}/s]"
                )

    elapsed = time.time() - t0
    logger.info(
        f"Done in {elapsed:.0f}s. "
        f"{success} downloaded, {skip} skipped, {fail} failed."
    )


def main():
    parser = argparse.ArgumentParser(description="Collect pre-1914 texts from Internet Archive")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/bellechat"),
                        help="Base cache directory")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <cache-dir>/ia_clean)")
    parser.add_argument("--max-items", type=int, default=50000,
                        help="Maximum items to process (default: 50000)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download workers (default: 8)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output = args.output or os.path.join(args.cache_dir, "internetarchive", "clean")
    collect(output, max_items=args.max_items, max_workers=args.workers)


if __name__ == "__main__":
    main()
