"""
Download and clean Project Gutenberg texts based on the pre-1914 manifest.

Reads the CSV manifest produced by gutenberg_rdf.py, downloads each text,
strips Gutenberg boilerplate, cleans the text, and saves to output directory.

Usage:
    # First, build the manifest:
    python -m data.gutenberg.gutenberg_rdf

    # Then download and clean:
    python -m data.gutenberg.collect [--workers 8] [--cache-dir DIR]
"""

import os
import csv
import time
import logging
import argparse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.common.cleaning import (
    normalize_unicode,
    collapse_whitespace,
    remove_gutenberg_headers,
    remove_illustration_tags,
    is_quality_text,
)

logger = logging.getLogger(__name__)

# Alternative URL patterns to try (Gutenberg uses inconsistent naming)
URL_PATTERNS = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
]


def download_text(ebook_id: int) -> str | None:
    """Download a Gutenberg ebook as plain text, trying multiple URL patterns."""
    for pattern in URL_PATTERNS:
        url = pattern.format(id=ebook_id)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bellechat/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                # Try UTF-8 first, fall back to latin-1
                raw = resp.read()
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError:
                    return raw.decode("latin-1")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            continue
    return None


def clean_text(raw_text: str) -> str:
    """Apply the full cleaning pipeline to a Gutenberg text."""
    text = remove_gutenberg_headers(raw_text)
    text = remove_illustration_tags(text)
    text = normalize_unicode(text)
    text = collapse_whitespace(text)
    return text


def process_one(row: dict, output_dir: str) -> tuple[bool, str]:
    """
    Download and clean a single ebook. Returns (success, message).
    Skips if output file already exists.
    """
    ebook_id = int(row["ebook_id"])
    output_path = os.path.join(output_dir, f"{ebook_id}.txt")

    if os.path.exists(output_path):
        return True, f"{ebook_id}: already exists"

    raw = download_text(ebook_id)
    if raw is None:
        return False, f"{ebook_id}: download failed"

    cleaned = clean_text(raw)

    if not is_quality_text(cleaned):
        return False, f"{ebook_id}: failed quality check (len={len(cleaned)})"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    return True, f"{ebook_id}: ok ({len(cleaned):,} chars)"


def collect(manifest_path: str, output_dir: str, max_workers: int = 8):
    """Download and clean all texts in the manifest."""
    os.makedirs(output_dir, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    logger.info(f"Manifest has {len(rows)} entries. Output: {output_dir}")
    logger.info(f"Using {max_workers} download workers")

    success_count = 0
    fail_count = 0
    skip_count = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_one, row, output_dir): row for row in rows}

        for i, future in enumerate(as_completed(futures)):
            ok, msg = future.result()
            if "already exists" in msg:
                skip_count += 1
            elif ok:
                success_count += 1
            else:
                fail_count += 1

            total_done = success_count + fail_count + skip_count
            if total_done % 500 == 0:
                elapsed = time.time() - t0
                rate = total_done / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {total_done}/{len(rows)} "
                    f"({success_count} ok, {fail_count} fail, {skip_count} skip) "
                    f"[{rate:.1f}/s]"
                )

    elapsed = time.time() - t0
    logger.info(
        f"Done in {elapsed:.0f}s. "
        f"{success_count} downloaded, {skip_count} skipped, {fail_count} failed."
    )


def main():
    parser = argparse.ArgumentParser(description="Download and clean Gutenberg pre-1914 texts")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/bellechat"),
                        help="Base cache directory")
    parser.add_argument("--manifest", default=None,
                        help="Path to manifest CSV (default: <cache-dir>/gutenberg_manifest.csv)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <cache-dir>/gutenberg_clean)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download workers (default: 8)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    manifest = args.manifest or os.path.join(args.cache_dir, "gutenberg", "manifest.csv")
    output = args.output or os.path.join(args.cache_dir, "gutenberg", "clean")

    if not os.path.exists(manifest):
        print(f"Manifest not found at {manifest}")
        print("Run 'python -m data.gutenberg.gutenberg_rdf' first to build it.")
        return

    collect(manifest, output, max_workers=args.workers)


if __name__ == "__main__":
    main()
