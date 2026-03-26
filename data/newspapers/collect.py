"""
Download and process Chronicling America newspaper OCR from LOC bulk archives.

Downloads tar.bz2 batch files from chroniclingamerica.loc.gov/ocr/, extracts
pre-1914 OCR text, applies quality filtering, and saves cleaned text.

Can process a single batch or fetch the full batch index and process all.

Usage:
    # Process a single batch file:
    python -m data.newspapers.collect --batch /path/to/batch.tar.bz2

    # Download and process batches from the LOC index:
    python -m data.newspapers.collect --download --max-batches 100

    # The run_all.sh script handles the full bulk download with parallelism.
"""

import os
import re
import json
import tarfile
import logging
import argparse
import unicodedata
import urllib.request

from data.common.cleaning import normalize_unicode, collapse_whitespace, estimate_ocr_quality

logger = logging.getLogger(__name__)

OCR_INDEX_URL = "https://chroniclingamerica.loc.gov/ocr.json"
MIN_OCR_QUALITY = 0.80
MIN_LENGTH = 200

# Load word set once for OCR quality checking
_WORD_SET = None


def _load_word_set() -> set[str]:
    global _WORD_SET
    if _WORD_SET is not None:
        return _WORD_SET
    dict_path = "/usr/share/dict/words"
    if os.path.exists(dict_path):
        with open(dict_path, "r") as f:
            _WORD_SET = set(w.strip().lower() for w in f if len(w.strip()) >= 2)
    else:
        logger.warning("No word dictionary found at /usr/share/dict/words")
        _WORD_SET = set()
    return _WORD_SET


def process_batch(tar_path: str, output_dir: str, processed_log: str | None = None):
    """
    Extract pre-1914 OCR text from a LOC batch tar.bz2 file.
    Applies date filtering, length check, and OCR quality filter.
    """
    os.makedirs(output_dir, exist_ok=True)
    word_set = _load_word_set()
    tar_name = os.path.basename(tar_path)

    saved = 0
    skipped_date = 0
    skipped_short = 0
    skipped_quality = 0
    total_chars = 0

    try:
        with tarfile.open(tar_path, "r:bz2") as tar:
            for member in tar:
                if not member.name.endswith("ocr.txt"):
                    continue

                # Date filter: year must be in path and <= 1913
                m = re.search(r"/(\d{4})/", member.name)
                if not m:
                    continue
                if int(m.group(1)) > 1913:
                    skipped_date += 1
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue
                text = f.read().decode("utf-8", errors="replace").strip()

                if len(text) < MIN_LENGTH:
                    skipped_short += 1
                    continue

                # OCR quality check
                quality = estimate_ocr_quality(text, word_set)
                if quality < MIN_OCR_QUALITY:
                    skipped_quality += 1
                    continue

                # Clean
                text = normalize_unicode(text)
                text = re.sub(r"[ \t]{3,}", "  ", text)
                text = collapse_whitespace(text)

                # Build filename from path
                safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", member.name.strip("/").replace("/", "_"))
                out_path = os.path.join(output_dir, safe_name)

                if not os.path.exists(out_path):
                    with open(out_path, "w", encoding="utf-8") as of:
                        of.write(text)
                    total_chars += len(text)
                    saved += 1
    except Exception as e:
        logger.error(f"Error processing {tar_name}: {e}")

    # Log this batch as processed
    if processed_log:
        with open(processed_log, "a") as f:
            f.write(tar_name + "\n")

    logger.info(
        f"{tar_name}: {saved} saved, {skipped_date} post-1913, "
        f"{skipped_short} short, {skipped_quality} low-OCR, {total_chars:,} chars"
    )
    return saved


def fetch_batch_index(cache_dir: str) -> list[dict]:
    """Fetch the LOC OCR batch index."""
    index_path = os.path.join(cache_dir, "ocr_index.json")
    logger.info("Fetching batch index from LOC...")
    urllib.request.urlretrieve(OCR_INDEX_URL, index_path)
    with open(index_path) as f:
        data = json.load(f)
    return data.get("ocr", [])


def load_processed(log_path: str) -> set[str]:
    """Load set of already-processed batch names."""
    if not os.path.exists(log_path):
        return set()
    with open(log_path) as f:
        return set(line.strip() for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser(description="Collect Chronicling America newspaper OCR text")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/bellechat"),
                        help="Base cache directory")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <cache-dir>/newspapers_clean)")
    parser.add_argument("--batch", default=None,
                        help="Process a single batch tar.bz2 file")
    parser.add_argument("--download", action="store_true",
                        help="Download and process batches from LOC index")
    parser.add_argument("--max-batches", type=int, default=10,
                        help="Maximum batches to download (default: 10)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output = args.output or os.path.join(args.cache_dir, "newspapers", "clean")
    raw_dir = os.path.join(args.cache_dir, "newspapers", "raw")
    processed_log = os.path.join(raw_dir, "processed_batches.txt")
    os.makedirs(raw_dir, exist_ok=True)

    if args.batch:
        # Process a single batch file
        process_batch(args.batch, output, processed_log)
    elif args.download:
        # Download and process from LOC index
        batches = fetch_batch_index(raw_dir)
        processed = load_processed(processed_log)
        remaining = [b for b in batches if b["name"] not in processed
                     and not any(x in b["name"] for x in ["vnstcsc", "prru"])]
        remaining.sort(key=lambda b: b["size"])

        logger.info(f"Total batches: {len(batches)}, already processed: {len(processed)}, "
                    f"remaining: {len(remaining)}")

        for i, batch in enumerate(remaining[:args.max_batches]):
            name = batch["name"]
            url = batch["url"]
            tar_path = os.path.join(raw_dir, name)

            logger.info(f"[{i+1}/{min(len(remaining), args.max_batches)}] Downloading {name}...")
            try:
                urllib.request.urlretrieve(url, tar_path)
            except Exception as e:
                logger.warning(f"Download failed: {e}")
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                continue

            if os.path.getsize(tar_path) < 10000:
                logger.warning(f"Blocked by Cloudflare, skipping {name}")
                os.remove(tar_path)
                with open(processed_log, "a") as f:
                    f.write(name + "\n")
                continue

            process_batch(tar_path, output, processed_log)
            os.remove(tar_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
