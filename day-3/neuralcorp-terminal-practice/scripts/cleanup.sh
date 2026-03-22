#!/usr/bin/env bash
# cleanup.sh — Archives old log files after a training run
#
# Usage: bash scripts/cleanup.sh <log_directory>
# Example: bash scripts/cleanup.sh data/

set -e   # stop immediately if any command fails (safety — always include this)

# Check that the user passed a directory argument
if [ "$#" -ne 1 ]; then
    echo "Usage: bash scripts/cleanup.sh <log_directory>" >&2
    exit 1   # exit code 1 = failure
fi

LOG_DIR="$1"   # named variable — much clearer than using $1 everywhere

# Verify the directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: '$LOG_DIR' is not a directory." >&2
    exit 1
fi

# Create a timestamped archive folder
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_DIR="${LOG_DIR}/archived_${TIMESTAMP}"
mkdir -p "$ARCHIVE_DIR"
echo "Created: $ARCHIVE_DIR"

# Move all .txt files into the archive
LOG_COUNT=0
for log_file in $(find "$LOG_DIR" -maxdepth 1 -name "*.txt" -type f); do
    mv "$log_file" "$ARCHIVE_DIR/"
    LOG_COUNT=$((LOG_COUNT + 1))
    echo "  Archived: $(basename $log_file)"
done

echo ""
echo "Done. Archived $LOG_COUNT log file(s)."