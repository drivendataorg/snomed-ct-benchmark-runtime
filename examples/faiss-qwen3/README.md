# FAISS + Qwen3 Example Submission

This is a working example submission that demonstrates a two-stage approach to SNOMED CT entity linking:

1. **Mention detection** using exact token matching against a FAISS index built from TF-IDF character n-gram vectors over SNOMED CT terminology.
2. **LLM disambiguation** using [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) to select the best candidate concept given clinical context and SNOMED CT metadata.

## Prerequisites

- **[uv](https://docs.astral.sh/uv/)** 
- **SNOMED CT RF2 release (International November 2025 Edition)**. Place the ZIP in `data/` or set the `SNOMED_RF2_ZIP` environment variable.
- **HuggingFace token** to download Qwen3-4B model weights. Set the `HF_TOKEN` environment variable or use the huggingface cli to run `hf auth login`
- **GPU with >=8GB VRAM (optional, recommended)**. Qwen3-4B runs on CPU but will be very slow. 

## Setup

### Quick start

```sh
just pack-example faiss-qwen3
```

`pack_submission.sh` handles everything: generating the terminology file (if missing), downloading model weights (if missing), and creating `submission/submission.zip`.

The script will look for the SNOMED CT RF2 ZIP in `data/` or at the path specified by `SNOMED_RF2_ZIP`.

### Manual steps

If you prefer to run each step individually:

#### 1. Generate the terminology file

The `flatten_terminology.py` script extracts and enriches concepts from SNOMED CT into a JSONL file that the submission uses at inference time. 

```sh
uv run examples/faiss-qwen3/flatten_terminology.py \
    --rf2-zip data/SnomedCT_InternationalRF2_PRODUCTION_*.zip
```

This produces `data/flattened_terminology.jsonl`. The script filters to concept types relevant to the benchmark (procedures, body structures, findings, etc.) and enriches each concept with synonyms, parent concepts, child concepts, and defining relationships.

#### 2. Pack the submission

```sh
just pack-example faiss-qwen3
```

This downloads Qwen3-4B weights into `model/` (if not already present), skips terminology generation (since the file already exists), and creates `submission/submission.zip`.

### 3. Create local test data

This step requires that you have downloaded `train_notes.csv` and `train_annotations.csv` from the [benchmark data repository](https://physionet.org/content/snomed-ct-entity-challenge/1.2.1) and placed them in the `data/` directory.

```sh
just smoke-test-data
```

This creates local `test_notes.csv` and `smoke_test_annotations.csv` files that should be identical to those contained in the benchmark smoke test environment. 

### 4. Test the submission

```sh
just test-submission
```

### 5. Score the generated predictions

```sh
uv run scripts/scoring.py \
    submission/predictions.csv \
    data/smoke_test_annotations.csv
```

## How it works

### Terminology index (`FaissTerminologyIndex`)

The index is built from `flattened_terminology.jsonl`. For each concept, the preferred name and all synonyms are normalized and vectorized using TF-IDF character n-grams (3-5 characters). The sparse vectors are reduced to 256 dimensions with TruncatedSVD and indexed in a FAISS inner-product index for fast cosine similarity search.

### Mention detection (`iter_mentions`)

Clinical notes are tokenized and all contiguous subsequences of up to 6 tokens are checked against the terminology index. Only spans that exactly match a normalized term are kept. Overlapping mentions are resolved by preferring longer, higher-confidence spans.

### LLM disambiguation (`LLMLinker`)

When multiple candidate concepts match a mention, the LLM is prompted with the mention text, surrounding clinical context (120 chars each side), and a formatted list of candidates including SNOMED CT metadata (hierarchy type, synonyms, parent concepts, child concepts, and defining relationships). The model generates a concept_id, which is extracted from the output.

## Ideas for improvement

- **Better mention detection.** Add fuzzy matching, edit distance, or a trained NER model (e.g., spaCy with a clinical model, BioBERT) to catch misspellings and abbreviations.
- **Better embeddings.** Replace TF-IDF with sentence-transformers or a domain-specific embedding model for the FAISS index.
- **Larger models.** Swap in a larger model or fine-tune on SNOMED CT entity linking examples.
- **Fine-tuning.** This example doesn't even use any of the training data! A straightforward next step would be to fine-tune on the training set for better disambiguation performance.
- **Richer terminology.** Enrich the JSONL with grandparent concepts, reference set memberships, or other SNOMED CT metadata by extending `flatten_terminology.py`.
- **Prompt tuning.** Add few-shot examples, adjust the context window, or restructure the prompt for better disambiguation.

## File overview

| File | Purpose |
|---|---|
| `main.py` | Submission entry point — runs mention detection and LLM disambiguation |
| `flatten_terminology.py` | Preprocessing script — generates `data/flattened_terminology.jsonl` from RF2 |
| `pack_submission.sh` | Generates terminology, downloads model, and creates `submission.zip` |
| `model/` | Downloaded model weights (git-ignored, created by `pack_submission.sh`) |
