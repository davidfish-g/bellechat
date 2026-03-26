#!/bin/bash
#
# Download and process ALL remaining Chronicling America newspaper batches.
# Runs 3 workers in parallel.
#
# Usage:
#   nohup bash data/newspapers/run_all.sh &
#   tail -f ~/.cache/bellechat/newspapers/raw/download.log
#
CACHE_DIR="$HOME/.cache/bellechat"
RAW_DIR="$CACHE_DIR/newspapers/raw"
LOG_FILE="$RAW_DIR/download.log"
PROCESSED_LOG="$RAW_DIR/processed_batches.txt"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKERS=3

mkdir -p "$RAW_DIR"
touch "$PROCESSED_LOG"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Activate venv
cd "$SCRIPT_DIR/../.."
source .venv/bin/activate

process_one() {
    local url="$1"
    local name=$(basename "$url")

    if grep -qF "$name" "$PROCESSED_LOG" 2>/dev/null; then
        return 0
    fi

    local tmpfile="$RAW_DIR/$name.$$"
    curl -sL --max-time 300 -o "$tmpfile" "$url"
    local size=$(stat -f%z "$tmpfile" 2>/dev/null || stat -c%s "$tmpfile" 2>/dev/null || echo 0)

    if [ "$size" -lt 10000 ]; then
        log "BLOCKED: $name ($size bytes)"
        rm -f "$tmpfile"
        echo "$name" >> "$PROCESSED_LOG"
        sleep 15
        return
    fi

    log "Processing $name ($((size / 1048576))MB)..."
    mv "$tmpfile" "$RAW_DIR/$name"
    python -m data.newspapers.collect --batch "$RAW_DIR/$name" 2>&1 | while read line; do
        log "  $line"
    done
    rm -f "$RAW_DIR/$name"
}

export -f process_one log
export CACHE_DIR RAW_DIR LOG_FILE PROCESSED_LOG

log "=== Starting newspaper download pipeline ($WORKERS parallel workers) ==="

# Fetch batch index and build URL list
log "Fetching batch index..."
curl -s "$( echo 'https://chroniclingamerica.loc.gov/ocr.json' )" > /tmp/ocr_index.json

python3 -c "
import json, os
with open('/tmp/ocr_index.json') as f:
    data = json.load(f)
batches = data.get('ocr', [])
processed = set()
log_path = '$PROCESSED_LOG'
if os.path.exists(log_path):
    with open(log_path) as f:
        processed = set(l.strip() for l in f if l.strip())
remaining = [b for b in batches if b['name'] not in processed
             and not any(x in b['name'] for x in ['vnstcsc','prru'])]
remaining.sort(key=lambda b: b['size'])
for b in remaining:
    print(b['url'])
" > /tmp/newspaper_remaining_urls.txt

TOTAL=$(wc -l < /tmp/newspaper_remaining_urls.txt | tr -d ' ')
log "Batches remaining: $TOTAL"

if command -v parallel &> /dev/null; then
    cat /tmp/newspaper_remaining_urls.txt | parallel -j "$WORKERS" process_one {}
else
    cat /tmp/newspaper_remaining_urls.txt | xargs -P "$WORKERS" -I {} bash -c 'process_one "$@"' _ {}
fi

log "=== COMPLETE ==="
