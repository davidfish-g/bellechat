"""
Convert cleaned text files from all data sources into nanochat-compatible parquet shards.

Reads from the three source directories (gutenberg_clean, britannica_clean,
newspapers_clean), shuffles all documents, and writes parquet shards matching
nanochat's expected format:
  - Single column named "text"
  - Row group size: 1024
  - Compression: zstd level 3
  - ~250M characters per shard
  - Last shard is reserved as the validation split

Usage:
    python -m data.sharding.shard_to_parquet [--cache-dir DIR]
"""

import os
import random
import time
import logging
import argparse

import pyarrow as pa
import pyarrow.parquet as pq

from data.common.cleaning import remove_modern_boilerplate

logger = logging.getLogger(__name__)

CHARS_PER_SHARD = 250_000_000
ROW_GROUP_SIZE = 1024
SEED = 42
VAL_FRACTION = 0.01  # 1% of documents held out for validation


def enumerate_documents(source_dirs: list[tuple[str, int]]) -> list[str]:
    """
    Enumerate all .txt files across source directories.
    Each entry is (directory_path, repeat_count).
    Returns a list of file paths (with repeats for upweighted sources).
    """
    all_paths = []
    for source_dir, repeat in source_dirs:
        if not os.path.isdir(source_dir):
            logger.warning(f"Source directory not found, skipping: {source_dir}")
            continue
        paths = [os.path.join(source_dir, fname)
                 for fname in os.listdir(source_dir) if fname.endswith(".txt")]
        logger.info(f"  {source_dir}: {len(paths):,} files x{repeat}")
        for _ in range(repeat):
            all_paths.extend(paths)
    return all_paths


def write_shard(docs: list[str], shard_path: str):
    """Write a list of document strings as a parquet shard."""
    table = pa.Table.from_pydict({"text": docs})
    pq.write_table(
        table,
        shard_path,
        row_group_size=ROW_GROUP_SIZE,
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
    )


def shard_documents(cache_dir: str):
    """Main sharding pipeline."""
    source_dirs = [
        (os.path.join(cache_dir, "gutenberg", "clean"), 1),
        (os.path.join(cache_dir, "britannica", "clean"), 5),  # 5x upweight for dense factual knowledge
        (os.path.join(cache_dir, "newspapers", "clean"), 1),
        (os.path.join(cache_dir, "internetarchive", "clean"), 1),
    ]
    output_dir = os.path.join(cache_dir, "shards")
    os.makedirs(output_dir, exist_ok=True)

    # Enumerate all documents
    logger.info("Enumerating documents from all sources...")
    doc_paths = enumerate_documents(source_dirs)
    logger.info(f"Found {len(doc_paths):,} documents total")

    if not doc_paths:
        logger.error("No documents found. Run data collection scripts first.")
        return

    # Shuffle
    rng = random.Random(SEED)
    rng.shuffle(doc_paths)

    # Split into train and validation
    val_count = max(1, int(len(doc_paths) * VAL_FRACTION))
    val_paths = doc_paths[:val_count]
    train_paths = doc_paths[val_count:]
    logger.info(f"Train: {len(train_paths):,} docs, Validation: {len(val_paths):,} docs")

    # Write training shards
    shard_docs = []
    shard_chars = 0
    shard_index = 0
    total_chars = 0
    t0 = time.time()

    for i, path in enumerate(train_paths):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            continue

        if not text.strip():
            continue

        text = remove_modern_boilerplate(text)
        if not text.strip():
            continue

        shard_docs.append(text)
        shard_chars += len(text)

        # Write shard when big enough and doc count is a multiple of row group size
        if shard_chars >= CHARS_PER_SHARD and len(shard_docs) % ROW_GROUP_SIZE == 0:
            shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
            write_shard(shard_docs, shard_path)

            dt = time.time() - t0
            total_chars += shard_chars
            logger.info(
                f"Shard {shard_index}: {len(shard_docs):,} docs, "
                f"{shard_chars:,} chars, {dt:.1f}s"
            )

            shard_docs = []
            shard_chars = 0
            shard_index += 1
            t0 = time.time()

    # Write any remaining train docs (pad to row group size with empty if needed? no — just write as-is)
    if shard_docs:
        shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        write_shard(shard_docs, shard_path)
        total_chars += shard_chars
        logger.info(
            f"Shard {shard_index} (final train): {len(shard_docs):,} docs, "
            f"{shard_chars:,} chars"
        )
        shard_index += 1

    # Write validation shard (always the last shard)
    val_docs = []
    val_chars = 0
    for path in val_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            continue
        text = remove_modern_boilerplate(text)
        if text.strip():
            val_docs.append(text)
            val_chars += len(text)

    if val_docs:
        val_shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        write_shard(val_docs, val_shard_path)
        total_chars += val_chars
        logger.info(
            f"Shard {shard_index} (validation): {len(val_docs):,} docs, "
            f"{val_chars:,} chars"
        )

    # Summary
    estimated_tokens = int(total_chars / 3.5)
    logger.info(f"\nSharding complete!")
    logger.info(f"  Total shards: {shard_index + 1} ({shard_index} train + 1 val)")
    logger.info(f"  Total characters: {total_chars:,}")
    logger.info(f"  Estimated tokens: {estimated_tokens:,}")
    logger.info(f"  Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert cleaned text to nanochat parquet shards")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/bellechat"),
                        help="Base cache directory containing *_clean subdirectories")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_documents(args.cache_dir)


if __name__ == "__main__":
    main()
