"""FAISS + Qwen3 entity linking submission for the SNOMED CT benchmark.

This solution works in two stages:
  1. **Mention detection** — scans clinical notes for text spans that exactly
     match normalized SNOMED CT terms, using a FAISS similarity index built
     from TF-IDF character n-gram vectors.
  2. **LLM disambiguation** — when multiple candidate concepts match a span,
     a causal language model (Qwen3-4B) selects the best concept using the
     surrounding clinical context and SNOMED CT metadata (hierarchy, synonyms,
     parent concepts, and defining relationships).

Places you might extend or improve this solution:
  - Swap in a different model by changing MODEL_DIR and updating the prompt.
  - Use vLLM for faster batched inference instead of HuggingFace generate().
  - Add fuzzy mention detection (e.g., edit distance, learned NER) alongside
    exact matching to catch misspellings or abbreviations.
  - Tune TOP_K_CANDIDATES, MAX_MENTION_TOKENS, or SIMILARITY_CUTOFF.
  - Enrich the terminology JSONL with more SNOMED CT metadata (grandparents,
    reference sets, etc.) via flatten_terminology.py.
  - Use sentence-transformers or a domain-specific embedding model instead of
    TF-IDF for the FAISS index.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import polars as pl
import torch
from loguru import logger
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths — these are relative to the code execution environment. During
# runtime, your submission is unpacked to /code_execution/src/ and data
# is mounted at /code_execution/data/.
# ---------------------------------------------------------------------------
NOTES_PATH = Path("data/test_notes.csv")
SUBMISSION_PATH = Path("submission.csv")
MODEL_DIR = Path(__file__).resolve().parent / "model"
TERMINOLOGY_PATH = Path(__file__).resolve().parent / "flattened_terminology.jsonl"

# ---------------------------------------------------------------------------
# Tunable parameters — these control the mention detection and candidate
# retrieval behavior. Adjusting these can significantly affect recall and
# precision.
# ---------------------------------------------------------------------------
# How many candidate concepts to retrieve per mention for LLM disambiguation.
TOP_K_CANDIDATES = 8
# Maximum number of whitespace-separated tokens in a mention span. Longer
# spans are more specific but may miss multi-word concepts.
MAX_MENTION_TOKENS = 6
# Minimum TF-IDF cosine similarity score for an exact-match candidate.
# Lower values increase recall at the cost of precision.
SIMILARITY_CUTOFF = 0.55

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'/-]*")
WHITESPACE_RE = re.compile(r"\s+")
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class Candidate:
    concept_id: str
    term: str
    score: float
    hierarchy: str
    synonyms: list[str]
    parents: list[dict[str, str]]
    children: list[dict[str, str]]
    relationships: list[dict[str, str]]


@dataclass(slots=True)
class Mention:
    start: int
    end: int
    text: str
    best_score: float
    candidates: list[Candidate]


def normalize_text(value: str) -> str:
    """Lowercase and strip non-alphanumeric characters for term matching."""
    collapsed = WHITESPACE_RE.sub(" ", value.strip().lower())
    normalized = NORMALIZE_RE.sub(" ", collapsed)
    return WHITESPACE_RE.sub(" ", normalized).strip()


def strip_hierarchy_suffix(value: str) -> str:
    """Remove the trailing parenthetical hierarchy type from a concept name,
    e.g., 'Laryngeal structure (body structure)' -> 'Laryngeal structure'."""
    return re.sub(r"\s+\([^()]*\)\s*$", "", value).strip()


def tokenize_with_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character offsets for each token in the text."""
    return [(match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


# ---------------------------------------------------------------------------
# FAISS terminology index
# ---------------------------------------------------------------------------
class FaissTerminologyIndex:
    """Semantic search index over SNOMED CT terminology.

    Builds a TF-IDF character n-gram vectorization of all concept terms and
    synonyms, reduces dimensionality with TruncatedSVD, and indexes the
    resulting dense vectors in a FAISS inner-product index for fast retrieval.

    You could replace TF-IDF with sentence-transformers or a domain-specific
    embedding model for potentially better retrieval quality.
    """

    def __init__(self, terminology_path: Path) -> None:
        logger.info("Loading terminology from {}", terminology_path)
        records: list[dict[str, object]] = []
        with terminology_path.open(encoding="utf-8") as terminology_file:
            for line in terminology_file:
                if not line.strip():
                    continue
                payload = json.loads(line)
                records.append(
                    {
                        "concept_id": str(payload["concept_id"]),
                        "concept_name": str(payload["concept_name"]),
                        "hierarchy": str(payload.get("hierarchy", "")),
                        "synonyms": payload.get("synonyms", []),
                        "parents": payload.get("parents", []),
                        "children": payload.get("children", []),
                        "relationships": payload.get("relationships", []),
                    }
                )

        terminology = pl.DataFrame(records)
        if terminology.is_empty():
            raise ValueError(f"Terminology file {terminology_path} is empty")
        logger.info("Loaded {:,} concepts from terminology", len(records))

        self.concept_ids = terminology["concept_id"].to_list()
        self.terms = terminology["concept_name"].to_list()
        self.hierarchies = terminology["hierarchy"].to_list()
        self.synonyms = terminology["synonyms"].to_list()
        self.parents = terminology["parents"].to_list()
        self.children = terminology["children"].to_list()
        self.relationships = terminology["relationships"].to_list()

        self.entry_concept_indices: list[int] = []
        self.search_terms: list[str] = []
        self.normalized_terms: list[str] = []
        self.exact_term_rows: dict[str, list[int]] = {}

        for concept_index, search_variants in enumerate(self._build_search_variants()):
            for search_term in search_variants:
                normalized_search_term = normalize_text(search_term)
                if not normalized_search_term:
                    continue
                self.entry_concept_indices.append(concept_index)
                self.search_terms.append(search_term)
                self.normalized_terms.append(normalized_search_term)
                self.exact_term_rows.setdefault(normalized_search_term, []).append(
                    concept_index
                )

        if not self.normalized_terms:
            raise ValueError(
                f"Terminology file {terminology_path} has no searchable terms"
            )
        logger.info(
            "Built {:,} search terms from {:,} concepts",
            len(self.normalized_terms),
            len(self.concept_ids),
        )

        self.max_term_tokens = min(
            MAX_MENTION_TOKENS,
            max(term.count(" ") + 1 for term in self.normalized_terms),
        )

        # Build TF-IDF vectors using character n-grams (3-5 chars). Character
        # n-grams are robust to morphological variation in medical terms.
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            lowercase=False,
            dtype=np.float32,
        )
        sparse_matrix = self.vectorizer.fit_transform(self.normalized_terms)
        max_components = min(
            256, sparse_matrix.shape[0] - 1, sparse_matrix.shape[1] - 1
        )
        self.reducer: TruncatedSVD | None
        if max_components >= 32:
            self.reducer = TruncatedSVD(n_components=max_components, random_state=0)
            dense_matrix = self.reducer.fit_transform(sparse_matrix).astype(np.float32)
        else:
            self.reducer = None
            dense_matrix = sparse_matrix.toarray().astype(np.float32)

        faiss.normalize_L2(dense_matrix)
        self.index = faiss.IndexFlatIP(dense_matrix.shape[1])
        self.index.add(dense_matrix)
        logger.info(
            "FAISS index ready — {} vectors, {} dimensions",
            self.index.ntotal,
            dense_matrix.shape[1],
        )

    def _build_search_variants(self) -> list[list[str]]:
        """For each concept, build a list of search terms from the preferred
        name and synonyms, stripping hierarchy suffixes."""
        all_variants: list[list[str]] = []
        for term, synonyms in zip(self.terms, self.synonyms, strict=False):
            seen: set[str] = set()
            variants: list[str] = []
            for raw_variant in [term, *synonyms]:
                variant = strip_hierarchy_suffix(str(raw_variant))
                if not variant or variant in seen:
                    continue
                variants.append(variant)
                seen.add(variant)
            all_variants.append(variants)
        return all_variants

    def _embed_queries(self, queries: list[str]) -> np.ndarray:
        query_matrix = self.vectorizer.transform(queries)
        if self.reducer is not None:
            dense_queries = self.reducer.transform(query_matrix).astype(np.float32)
        else:
            dense_queries = query_matrix.toarray().astype(np.float32)
        faiss.normalize_L2(dense_queries)
        return dense_queries

    def search(
        self, mention_text: str, top_k: int = TOP_K_CANDIDATES
    ) -> list[Candidate]:
        """Search for candidate concepts matching a mention text span.

        Returns up to top_k candidates sorted by descending similarity score.
        Exact term matches are boosted to score 1.0.
        """
        normalized_query = normalize_text(mention_text)
        if not normalized_query:
            return []

        dense_query = self._embed_queries([normalized_query])
        search_k = min(max(top_k * 8, top_k), len(self.entry_concept_indices))
        scores, indices = self.index.search(dense_query, search_k)

        deduped: dict[str, Candidate] = {}

        # Exact matches get a perfect score
        for concept_index in self.exact_term_rows.get(normalized_query, []):
            candidate = Candidate(
                concept_id=self.concept_ids[concept_index],
                term=self.terms[concept_index],
                score=1.0,
                hierarchy=self.hierarchies[concept_index],
                synonyms=self.synonyms[concept_index],
                parents=self.parents[concept_index],
                children=self.children[concept_index],
                relationships=self.relationships[concept_index],
            )
            deduped[candidate.concept_id] = candidate

        # FAISS approximate nearest neighbor results
        for score, entry_index in zip(scores[0], indices[0], strict=False):
            if entry_index < 0:
                continue
            concept_index = self.entry_concept_indices[entry_index]
            candidate = Candidate(
                concept_id=self.concept_ids[concept_index],
                term=self.terms[concept_index],
                score=float(score),
                hierarchy=self.hierarchies[concept_index],
                synonyms=self.synonyms[concept_index],
                parents=self.parents[concept_index],
                children=self.children[concept_index],
                relationships=self.relationships[concept_index],
            )
            existing = deduped.get(candidate.concept_id)
            if existing is None or candidate.score > existing.score:
                deduped[candidate.concept_id] = candidate

        return sorted(
            deduped.values(),
            key=lambda candidate: (-candidate.score, candidate.term),
        )[:top_k]

    def contains_exact_term(self, mention_text: str) -> bool:
        return normalize_text(mention_text) in self.exact_term_rows


# ---------------------------------------------------------------------------
# LLM-based concept disambiguation
# ---------------------------------------------------------------------------
class LLMLinker:
    """Uses a causal language model to select the best SNOMED CT concept from
    a list of candidates, given the clinical context around a mention.

    The model is loaded lazily on first use. If no GPU is available, it falls
    back to CPU with float32 (slower but functional).

    To swap in a different model:
      1. Download it into the model/ directory (or change MODEL_DIR).
      2. Adjust the prompt in choose_concept_id() if needed.
      3. Consider using vLLM for batched inference if throughput matters.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForCausalLM | None = None

    def _ensure_loaded(self) -> None:
        if not self.enabled:
            return
        if self.tokenizer is not None and self.model is not None:
            return

        logger.info("Loading LLM from {} (device={})", MODEL_DIR, self.device)
        model_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
            torch_dtype=model_dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("LLM loaded successfully")

    def choose_concept_id(
        self,
        note_text: str,
        mention: Mention,
        candidates: list[Candidate],
    ) -> str:
        """Select the best concept_id from candidates for a given mention.

        The prompt provides the LLM with:
          - The mention text and surrounding context
          - A formatted list of candidate concepts with SNOMED CT metadata
          - Background knowledge about SNOMED CT structure

        You could improve this by:
          - Tuning the prompt and context window size
          - Adding few-shot examples from training data
          - Using a larger or fine-tuned model
          - Implementing a confidence threshold to fall back to the top
            candidate when the LLM is uncertain
        """
        if not candidates:
            return ""
        if not self.enabled:
            return candidates[0].concept_id

        self._ensure_loaded()
        assert self.tokenizer is not None
        assert self.model is not None

        candidate_lines = "\n".join(
            self._format_candidate(candidate) for candidate in candidates
        )
        context_window = 120
        snippet_start = max(0, mention.start - context_window)
        snippet_end = min(len(note_text), mention.end + context_window)
        snippet = note_text[snippet_start:snippet_end]
        highlighted = (
            note_text[snippet_start : mention.start]
            + "[["
            + note_text[mention.start : mention.end]
            + "]]"
            + note_text[mention.end : snippet_end]
        )
        instructions = """## Using SNOMED CT for entity linking

Unlike simple medical coding systems, SNOMED CT has been carefully curated to capture and represent medical knowledge and practice. By exploring the attributes of concepts and the relationships between them, you can arm your models with prior knowledge about the clinical terms they will encounter. Below are examples demonstrating how this knowledge can be extracted.

## Terms
Every concept in SNOMED CT has a "fully specified name" (FSN). This is a descriptive term that is unique to the concept within the terminology. For example, the FSN of the concept:

4596009 |Laryngeal structure (body structure)|

is "Laryngeal structure (body structure)". The FSN does not always use natural language as it intends to closely match the logical definition of the concept.

SNOMED CT also records other terms ("synonyms") for each concept. These are alternative descriptions that have been deemed medically acceptable. For the above concept the synonyms are: "Laryngeal structure", "Larynx structure" and "Larynx". The presence of multiple descriptive terms for each concept is useful for training vector embeddings or simply for string-matching.

## Hierarchy

Every concept in SNOMED CT exists somewhere in the concept hierarchy. SNOMED CT uses an attribute called "is a" to define the parent-child relationships. The parents and children of a concept can help us to understand it in the proper context. For example, the concept:

4596009 |Laryngeal structure (body structure)|

has the following ("is a") parents:

49928004 |Structure of anterior portion of neck (body structure)|
303417002 |Larynx and/or tracheal structures (body structure)|
716151000 |Structure of oropharynx and/or hypopharynx and/or larynx (body structure)|
714323000 |Structure of organ in respiratory system (body structure)|
From this we learn that the concept in question is a body structure, that it forms part of the respiratory system, that it is located towards the front of the neck and that it is closely linked to the trachea. A concept's grandparents and further ancestors can be used to discover broader information about a concept. As we navigate up the hierarchy, each "is a" relationship is like a logical statement about the target concept.

The terminology also contains concepts and other content marked as inactive, but the use of these inactive concepts is outside the scope of this benchmark.

## Defining relationships

Every concept is modelled with zero or more sets of defining relationships with each relationship taking the form of an (attribute, value) pair. Together with the "is a" relationships, these sets constitute a unique (both necessary and sufficient) definition for a concept.

Consider the concept:

274317003 |Laryngoscopic biopsy larynx (procedure)|

Its defining relationships are:

Attribute	Value
260686004 |Method (attribute)|	129314006 |Biopsy - action (qualifier value)|
405813007 |Procedure site - Direct (attribute)|	4596009 |Laryngeal structure (body structure)|
425391005 |Using access device (attribute)|	44738004 |Laryngoscope, device (physical object)|


Suppose that an algorithm encountered the sentence: "Patient referred for a biopsy to investigate potential swelling in upper larynx". The algorithm matches the span "larynx" to 4596009 |Laryngeal structure (body structure)| and is now considering whether to match the span "biopsy" to a concept.

An obvious choice might be 86273004 |Biopsy (procedure)|, but we also have the semantically similar 129314006 |Biopsy - action (qualifier value)|. By using the defining relationships above – combined with the presence of the span "larynx" in close proximity to the span "biopsy" – the algorithm could make a pretty good guess that the relevant concept here is 274317003 |Laryngoscopic biopsy larynx (procedure)| – which would be the correct annotation in this instance.
"""
        messages = [
            {
                "role": "user",
                "content": (
                    "Link the highlighted clinical mention to the best SNOMED CT concept. Use the instructions below to inform your decision:\n"
                    f"{instructions.strip()}\n"
                    "I will provide you with a mention and its context. Return only one concept_id from the candidate list.\n"
                    f"Mention text: {mention.text}\n"
                    f"Context snippet: {snippet}\n"
                    f"Highlighted snippet: {highlighted}\n"
                    f"Candidates:\n{candidate_lines}"
                ),
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        for candidate in candidates:
            if re.search(rf"\b{re.escape(candidate.concept_id)}\b", output):
                return candidate.concept_id
        return candidates[0].concept_id

    @staticmethod
    def _format_candidate(candidate: Candidate) -> str:
        """Format a candidate concept for the LLM prompt, including SNOMED CT
        metadata that helps the model disambiguate."""
        details = [f"name={candidate.term}"]
        if candidate.hierarchy:
            details.append(f"hierarchy={candidate.hierarchy}")

        if candidate.synonyms:
            details.append(f"synonyms={'; '.join(candidate.synonyms[:5])}")

        if candidate.parents:
            parent_names = [
                parent.get("concept_name", "").strip()
                for parent in candidate.parents[:3]
                if parent.get("concept_name")
            ]
            if parent_names:
                details.append(f"parents={'; '.join(parent_names)}")

        if candidate.children:
            child_names = [
                child.get("concept_name", "").strip()
                for child in candidate.children[:3]
                if child.get("concept_name")
            ]
            if child_names:
                details.append(f"children={'; '.join(child_names)}")

        if candidate.relationships:
            relationship_pairs = []
            for relationship in candidate.relationships[:4]:
                type_name = str(relationship.get("type_name", "")).strip()
                destination_name = str(relationship.get("destination_name", "")).strip()
                if type_name and destination_name:
                    relationship_pairs.append(f"{type_name} -> {destination_name}")
            if relationship_pairs:
                details.append(f"relationships={'; '.join(relationship_pairs)}")

        return f"- {candidate.concept_id}: " + " | ".join(details)


# ---------------------------------------------------------------------------
# Mention detection
# ---------------------------------------------------------------------------
def iter_mentions(
    text: str, terminology_index: FaissTerminologyIndex
) -> list[Mention]:
    """Detect SNOMED CT mentions in clinical text using exact token matching.

    Scans all contiguous token subsequences (up to MAX_MENTION_TOKENS long)
    and checks whether each normalized span exists in the terminology index.
    Overlapping mentions are resolved by preferring longer, higher-confidence
    spans.

    This is a simple but effective approach. You could extend it by:
      - Adding fuzzy matching (edit distance, phonetic similarity)
      - Using a trained NER model (e.g., spaCy, BioBERT) to propose spans
      - Combining multiple mention detection strategies
    """
    token_spans = tokenize_with_spans(text)
    mentions: list[Mention] = []

    for token_index in range(len(token_spans)):
        max_length = min(
            terminology_index.max_term_tokens, len(token_spans) - token_index
        )
        for token_length in range(max_length, 0, -1):
            start = token_spans[token_index][0]
            end = token_spans[token_index + token_length - 1][1]
            mention_text = text[start:end]
            if not terminology_index.contains_exact_term(mention_text):
                continue

            candidates = terminology_index.search(mention_text)
            if not candidates:
                continue
            if candidates[0].score < SIMILARITY_CUTOFF:
                continue
            mentions.append(
                Mention(
                    start=start,
                    end=end,
                    text=mention_text,
                    best_score=candidates[0].score,
                    candidates=candidates,
                )
            )

    # Prefer longer, higher-confidence matches and then restore note order.
    mentions.sort(
        key=lambda m: (-(m.end - m.start), -m.best_score, m.start),
    )
    selected: list[Mention] = []
    occupied: set[int] = set()
    for mention in mentions:
        span_range = range(mention.start, mention.end)
        if any(i in occupied for i in span_range):
            continue
        selected.append(mention)
        occupied.update(span_range)
    return sorted(selected, key=lambda m: (m.start, m.end))


# ---------------------------------------------------------------------------
# Budget control
# ---------------------------------------------------------------------------
def max_model_invocations() -> int | None:
    """Determine the maximum number of LLM calls to make.

    Set the MAX_LLM_LINKS environment variable to limit LLM calls (useful
    for testing). During smoke tests without a GPU, the LLM is disabled
    entirely to allow fast validation of the pipeline.
    """
    override = os.environ.get("MAX_LLM_LINKS")
    if override:
        return int(override)
    if os.environ.get("IS_SMOKE_TEST") == "1" and not torch.cuda.is_available():
        return 0
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    terminology_index = FaissTerminologyIndex(TERMINOLOGY_PATH)
    llm_budget = max_model_invocations()
    linker = LLMLinker(enabled=llm_budget != 0)
    if llm_budget is not None:
        logger.info("LLM call budget: {}", llm_budget)
    else:
        logger.info("LLM call budget: unlimited")
    llm_calls = 0
    predictions: list[list[object]] = []

    # Read all notes up front so we can show a progress bar
    logger.info("Reading notes from {}", NOTES_PATH)
    with NOTES_PATH.open(newline="", encoding="utf-8") as notes_file:
        reader = csv.DictReader(notes_file)
        notes = list(reader)
    logger.info("Loaded {} notes", len(notes))

    for row in tqdm(notes, desc="Processing notes"):
        note_id = str(row["note_id"])
        note_text = str(row["text"])
        mentions = iter_mentions(note_text, terminology_index)

        for mention in mentions:
            if llm_budget is not None and llm_calls >= llm_budget:
                concept_id = mention.candidates[0].concept_id
            else:
                concept_id = linker.choose_concept_id(
                    note_text=note_text,
                    mention=mention,
                    candidates=mention.candidates,
                )
                if linker.enabled:
                    llm_calls += 1
            predictions.append([note_id, mention.start, mention.end, concept_id])

    logger.info(
        "Generated {:,} predictions ({:,} LLM calls)", len(predictions), llm_calls
    )
    with SUBMISSION_PATH.open("w", newline="", encoding="utf-8") as submission_file:
        writer = csv.writer(submission_file)
        writer.writerow(["note_id", "start", "end", "concept_id"])
        writer.writerows(predictions)
    logger.info("Submission written to {}", SUBMISSION_PATH)


if __name__ == "__main__":
    main()
