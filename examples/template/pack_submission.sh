#!/usr/bin/env bash
# Pack this example into a submission.zip file
# Usage: ./pack_submission.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/submission"

if [[ -f "$OUTPUT_DIR/submission.zip" ]]; then
    echo "Error: $OUTPUT_DIR/submission.zip already exists. Remove it first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$SCRIPT_DIR"
zip -r "$OUTPUT_DIR/submission.zip" ./* -x "pack_submission.sh"

echo "Created $OUTPUT_DIR/submission.zip"
