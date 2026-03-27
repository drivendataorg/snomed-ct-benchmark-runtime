#!/usr/bin/env bash
# Download model weights, generate terminology, and pack into submission.zip.
#
# Prerequisites:
#   - uv installed (https://docs.astral.sh/uv/)
#   - hf CLI installed (pip install huggingface_hub[hf_xet])
#   - HF_TOKEN environment variable set, or logged in via `hf auth login`
#   - SNOMED CT RF2 release ZIP (set SNOMED_RF2_ZIP or place in data/)
#
# Usage: ./pack_submission.sh <output_dir>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/${1:?Usage: $0 <output_dir>}"
MODEL_DIR="$SCRIPT_DIR/model"
TERMINOLOGY_FILE="$REPO_ROOT/data/flattened_terminology.jsonl"

# The HuggingFace model repo to download. Change this to use a different model.
HF_MODEL_REPO="Qwen/Qwen3-4B-Instruct-2507"

# ── Generate flattened_terminology.jsonl if missing ───────────────────────
if [[ -f "$TERMINOLOGY_FILE" ]]; then
    echo "Terminology file already exists at $TERMINOLOGY_FILE — skipping generation."
else
    # Find the RF2 ZIP: use SNOMED_RF2_ZIP env var, or glob for it in data/
    if [[ -n "${SNOMED_RF2_ZIP:-}" ]]; then
        RF2_ZIP="$SNOMED_RF2_ZIP"
    else
        RF2_ZIP=""
        for f in "$REPO_ROOT"/data/SnomedCT_*RF2*.zip; do
            if [[ -f "$f" ]]; then
                RF2_ZIP="$f"
                break
            fi
        done
    fi

    if [[ -z "$RF2_ZIP" ]]; then
        echo "Error: No SNOMED CT RF2 ZIP found."
        echo ""
        echo "Either:"
        echo "  1. Set SNOMED_RF2_ZIP=/path/to/SnomedCT_*.zip"
        echo "  2. Place the RF2 ZIP in data/"
        echo "  3. Generate the terminology manually:"
        echo "     uv run examples/faiss-qwen3/flatten_terminology.py --rf2-zip /path/to/SnomedCT_*.zip"
        echo ""
        echo "See README.md for details on obtaining the SNOMED CT RF2 release."
        exit 1
    fi

    echo "Generating terminology from $RF2_ZIP ..."
    uv run "$SCRIPT_DIR/flatten_terminology.py" --rf2-zip "$RF2_ZIP"
    echo ""
fi

# ── Download model weights from HuggingFace ───────────────────────────────
if [[ -d "$MODEL_DIR" ]] && [[ -n "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]]; then
    echo "Model directory already exists at $MODEL_DIR — skipping download."
    echo "To re-download, remove the model/ directory first."
else
    echo "Downloading model weights from $HF_MODEL_REPO ..."
    echo "This requires a HuggingFace token. Set HF_TOKEN or run 'hf auth login'."
    echo ""
    # Download to HF cache first (no-op if already cached), then copy to
    # the local directory. This lets multiple examples share the same cache
    # and avoids re-downloading if the model is already present.
    SNAPSHOT_DIR=$(hf download "$HF_MODEL_REPO")
    mkdir -p "$MODEL_DIR"
    cp -aL "$SNAPSHOT_DIR/." "$MODEL_DIR"
    echo "Model copied to $MODEL_DIR (from HF cache)"
fi

# ── Pack submission ───────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# Zip the example directory contents (main.py, model/) excluding dev files,
# then add the terminology file from data/ at the zip root (alongside main.py).
# Use -0 (store, no compression) since model weights are already compressed
# and re-compressing ~8 GB of safetensors is slow for no benefit.
(cd "$SCRIPT_DIR" && zip -0 -r "$OUTPUT_DIR/submission.zip" . \
    -x "./pack_submission.sh" \
    -x "./flatten_terminology.py" \
    -x "./README.md" \
    -x "./.gitignore")
zip -j "$OUTPUT_DIR/submission.zip" "$TERMINOLOGY_FILE"

echo ""
echo "Created $OUTPUT_DIR/submission.zip"
echo "Test it with: just test-submission"
