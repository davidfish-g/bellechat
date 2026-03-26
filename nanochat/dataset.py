"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import pyarrow.parquet as pq

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# bellechat: pretraining data is generated locally, not downloaded from HuggingFace.
# Run data/sharding/shard_to_parquet.py to generate shards from collected pre-1914 text.
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "shards")

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir

    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Run 'python -m data.sharding.shard_to_parquet' to generate shards from collected pre-1914 text."
        )

    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

if __name__ == "__main__":
    print("bellechat uses locally-generated pretraining data from pre-1914 text.")
    print(f"Expected data directory: {DATA_DIR}")
    print()
    print("To generate shards, run:")
    print("  python -m data.sharding.shard_to_parquet")
    if os.path.exists(DATA_DIR):
        files = list_parquet_files()
        print(f"\nFound {len(files)} parquet shards in {DATA_DIR}")
    else:
        print(f"\nData directory does not exist yet.")
