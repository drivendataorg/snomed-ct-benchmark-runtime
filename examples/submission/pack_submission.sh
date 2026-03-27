#!/usr/bin/env bash
# Pack this example into a submission.zip file.
# Usage: ./pack_submission.sh <output_dir>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/${1:?Usage: $0 <output_dir>}"

mkdir -p "$OUTPUT_DIR"
(cd "$SCRIPT_DIR" && zip -r "$OUTPUT_DIR/submission.zip" . -x "./pack_submission.sh")

echo "Created $OUTPUT_DIR/submission.zip"
