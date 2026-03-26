#!/bin/bash
#
# Download ALL pre-1914 Internet Archive texts in batches of 100K.
# Run with: nohup bash data/internetarchive/run_all.sh &
# Monitor with: tail -f ~/.cache/bellechat/internetarchive/clean/download.log
#
set -e

CACHE_DIR="$HOME/.cache/bellechat"
IA_DIR="$CACHE_DIR/internetarchive/clean"
LOG_FILE="$IA_DIR/download.log"
BATCH_SIZE=100000
WORKERS=8

# Activate venv
cd "$(dirname "$0")/../.."
source .venv/bin/activate

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

# Total available: ~2.24M items. Process in batches of 100K.
# The script tracks processed IDs in ia_clean/processed_ids.txt,
# so each run skips already-done items automatically.

ROUND=1
while true; do
    PROCESSED=$(wc -l < "$IA_DIR/processed_ids.txt" 2>/dev/null | tr -d ' ' || echo 0)
    IA_FILES=$(ls "$IA_DIR"/*.txt 2>/dev/null | wc -l | tr -d ' ')
    IA_SIZE=$(du -sh "$IA_DIR" 2>/dev/null | cut -f1)

    log "=== Round $ROUND: $PROCESSED IDs processed, $IA_FILES files, $IA_SIZE on disk ==="

    # Increase max-items each round to reach deeper into the catalog
    MAX_ITEMS=$((BATCH_SIZE * ROUND))
    if [ "$MAX_ITEMS" -gt 2300000 ]; then
        MAX_ITEMS=2300000
    fi

    log "Searching up to $MAX_ITEMS items..."
    python -m data.internetarchive.collect \
        --max-items "$MAX_ITEMS" \
        --workers "$WORKERS" \
        2>&1 | tee -a "$LOG_FILE"

    NEW_PROCESSED=$(wc -l < "$IA_DIR/processed_ids.txt" 2>/dev/null | tr -d ' ' || echo 0)
    GAINED=$((NEW_PROCESSED - PROCESSED))
    log "Round $ROUND complete. Processed $GAINED new items."

    # If we gained very few new items, we've exhausted the search results
    if [ "$GAINED" -lt 100 ]; then
        log "Fewer than 100 new items — likely exhausted available data. Stopping."
        break
    fi

    # If we've hit 2.3M processed, we're done
    if [ "$NEW_PROCESSED" -ge 2200000 ]; then
        log "Reached 2.2M+ processed items. Done!"
        break
    fi

    ROUND=$((ROUND + 1))
done

FINAL_FILES=$(ls "$IA_DIR"/*.txt 2>/dev/null | wc -l | tr -d ' ')
FINAL_SIZE=$(du -sh "$IA_DIR" 2>/dev/null | cut -f1)
log "=== ALL DONE: $FINAL_FILES files, $FINAL_SIZE ==="
