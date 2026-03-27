# /// script
# requires-python = ">=3.12"
# dependencies = ["polars", "loguru", "tqdm"]
# ///
"""Flatten SNOMED CT RF2 snapshot data into an enriched JSONL terminology file.

This script extracts concepts, descriptions, and relationships from an RF2
release ZIP and produces a single JSONL file where each line is a concept with
its preferred name, hierarchy type, synonyms, parent concepts, child concepts,
and defining relationships. The output is consumed by main.py at inference time.

Usage:
    uv run flatten_terminology.py --rf2-zip path/to/SnomedCT_*.zip

The RF2 ZIP can be downloaded from the SNOMED CT member licensing portal
or through UMLS after accepting the SNOMED CT license.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import zipfile
from pathlib import Path

import polars as pl
from loguru import logger
from tqdm import tqdm

csv.field_size_limit(10**9)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The concept types below define the subset of SNOMED CT concepts that are
# relevant to this benchmark. You can add or remove types here to change
# which concepts appear in the terminology file.
CONCEPT_TYPE_SUBSET = [
    "procedure",
    "body structure",
    "finding",
    "disorder",
    "morphologic abnormality",
    "regime/therapy",
    "cell structure",
]

# SNOMED CT description type IDs
PREFERRED_TYPE_ID = "900000000000003001"
ACCEPTABLE_TYPE_ID = "900000000000013009"

# The "is a" relationship type ID identifies parent-child relationships
IS_A_TYPE_ID = "116680003"


# ---------------------------------------------------------------------------
# RF2 file extraction
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten SNOMED CT RF2 snapshot into an enriched JSONL terminology file.",
    )
    parser.add_argument(
        "--rf2-zip",
        type=Path,
        required=True,
        help="Path to the SNOMED CT International RF2 release ZIP file.",
    )
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "flattened_terminology.jsonl",
        help="Output JSONL file path (default: data/flattened_terminology.jsonl in the repo root).",
    )
    return parser.parse_args()


def find_snapshot_member(members: list[str], prefix: str) -> str:
    matches = [
        member for member in members if prefix in member and member.endswith(".txt")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one ZIP member matching {prefix!r}, found {len(matches)}"
        )
    return matches[0]


def extract_snapshot_files(rf2_zip: Path, temp_dir: Path) -> tuple[Path, Path, Path]:
    logger.info("Extracting snapshot files from {}", rf2_zip.name)
    with zipfile.ZipFile(rf2_zip) as archive:
        members = archive.namelist()
        concept_member = find_snapshot_member(
            members, "/Snapshot/Terminology/sct2_Concept_Snapshot_"
        )
        description_member = find_snapshot_member(
            members,
            "/Snapshot/Terminology/sct2_Description_Snapshot-en_",
        )
        relationship_member = find_snapshot_member(
            members,
            "/Snapshot/Terminology/sct2_Relationship_Snapshot_",
        )
        concept_path = Path(archive.extract(concept_member, path=temp_dir))
        description_path = Path(archive.extract(description_member, path=temp_dir))
        relationship_path = Path(archive.extract(relationship_member, path=temp_dir))
    return concept_path, description_path, relationship_path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_active_concepts(concept_path: Path) -> pl.DataFrame:
    logger.info("Loading active concepts...")
    concept_ids: list[str] = []
    with concept_path.open(newline="", encoding="utf-8") as concept_file:
        reader = csv.DictReader(concept_file, delimiter="\t")
        for row in reader:
            if row["active"] != "1":
                continue
            concept_ids.append(row["id"])
    logger.info("Loaded {:,} active concepts", len(concept_ids))
    return pl.DataFrame({"concept_id": concept_ids})


def load_active_descriptions(description_path: Path) -> pl.DataFrame:
    logger.info("Loading active descriptions...")
    concept_ids: list[str] = []
    concept_names: list[str] = []
    name_type_ids: list[str] = []

    with description_path.open(newline="", encoding="utf-8") as description_file:
        reader = csv.DictReader(description_file, delimiter="\t")
        for row in reader:
            if row["active"] != "1":
                continue
            concept_ids.append(row["conceptId"])
            concept_names.append(row["term"])
            name_type_ids.append(row["typeId"])

    logger.info("Loaded {:,} active descriptions", len(concept_ids))
    return pl.DataFrame(
        {
            "concept_id": concept_ids,
            "concept_name": concept_names,
            "name_type_id": name_type_ids,
        }
    )


def load_active_relationships(relationship_path: Path) -> pl.DataFrame:
    logger.info("Loading active relationships...")
    source_ids: list[str] = []
    destination_ids: list[str] = []
    type_ids: list[str] = []

    with relationship_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["active"] != "1":
                continue
            source_ids.append(row["sourceId"])
            destination_ids.append(row["destinationId"])
            type_ids.append(row["typeId"])

    logger.info("Loaded {:,} active relationships", len(source_ids))
    return pl.DataFrame(
        {
            "source_id": source_ids,
            "destination_id": destination_ids,
            "type_id": type_ids,
        }
    )


# ---------------------------------------------------------------------------
# Terminology building
# ---------------------------------------------------------------------------
def build_flat_terminology(
    concepts: pl.DataFrame, descriptions: pl.DataFrame
) -> pl.DataFrame:
    """Build a flat terminology DataFrame containing preferred names for the
    concept type subset.

    This filters to concepts whose fully-specified name ends with a
    parenthetical hierarchy type matching CONCEPT_TYPE_SUBSET (e.g.,
    "(body structure)", "(procedure)"). You could expand the subset by
    adding more types to CONCEPT_TYPE_SUBSET above.
    """
    logger.info("Building flat terminology for subset: {}", CONCEPT_TYPE_SUBSET)
    terminology = (
        concepts.join(descriptions, on="concept_id", how="inner")
        .with_columns(
            pl.when(pl.col("name_type_id") == PREFERRED_TYPE_ID)
            .then(pl.lit("P"))
            .when(pl.col("name_type_id") == ACCEPTABLE_TYPE_ID)
            .then(pl.lit("A"))
            .otherwise(pl.lit(None))
            .alias("name_type"),
            pl.col("concept_name")
            .str.extract(r"\(([^()]*)\)\s*$", group_index=1)
            .str.to_lowercase()
            .alias("hierarchy"),
        )
        .filter(pl.col("name_type").is_in(["P", "A"]))
        .filter(pl.col("hierarchy").is_in(CONCEPT_TYPE_SUBSET))
        .filter(pl.col("name_type") == "P")
        .select("concept_id", "concept_name", "hierarchy")
        .sort(["hierarchy", "concept_id"])
    )
    logger.info("Terminology contains {:,} concepts", terminology.height)
    return terminology


def build_preferred_name_lookup(
    concepts: pl.DataFrame, descriptions: pl.DataFrame
) -> dict[str, str]:
    """Map concept_id -> preferred term for ALL active concepts (used to resolve
    relationship destinations and type names that may fall outside the hierarchy
    subset)."""
    logger.info("Building preferred name lookup...")
    preferred = (
        concepts.join(descriptions, on="concept_id", how="inner")
        .filter(pl.col("name_type_id") == PREFERRED_TYPE_ID)
        .select("concept_id", "concept_name")
    )
    return dict(
        zip(preferred["concept_id"].to_list(), preferred["concept_name"].to_list())
    )


def build_synonym_lookup(
    terminology: pl.DataFrame,
    descriptions: pl.DataFrame,
) -> dict[str, list[str]]:
    """Map concept_id -> list of acceptable synonym terms for concepts in the
    subset. Synonyms provide alternative surface forms for string matching and
    search index construction."""
    logger.info("Building synonym lookup...")
    concept_ids_in_subset = terminology["concept_id"].to_list()
    preferred_name_lookup = dict(
        zip(
            terminology["concept_id"].to_list(),
            terminology["concept_name"].to_list(),
        )
    )

    synonym_rows = (
        descriptions.filter(pl.col("concept_id").is_in(concept_ids_in_subset))
        .with_columns(
            pl.when(pl.col("name_type_id") == PREFERRED_TYPE_ID)
            .then(pl.lit("P"))
            .when(pl.col("name_type_id") == ACCEPTABLE_TYPE_ID)
            .then(pl.lit("A"))
            .otherwise(pl.lit(None))
            .alias("name_type")
        )
        .filter(pl.col("name_type") == "A")
        .select("concept_id", "concept_name")
        .sort(["concept_id", "concept_name"])
        .unique(
            subset=["concept_id", "concept_name"], keep="first", maintain_order=True
        )
    )

    synonyms_by_concept: dict[str, list[str]] = {}
    for concept_id, synonym in zip(
        synonym_rows["concept_id"].to_list(),
        synonym_rows["concept_name"].to_list(),
    ):
        if synonym == preferred_name_lookup.get(concept_id):
            continue
        synonyms_by_concept.setdefault(concept_id, []).append(synonym)
    return synonyms_by_concept


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------
def build_enriched_jsonl(
    terminology: pl.DataFrame,
    relationships: pl.DataFrame,
    name_lookup: dict[str, str],
    synonyms_lookup: dict[str, list[str]],
    output_path: Path,
) -> int:
    """Write an enriched JSONL file with one line per concept.

    Each line contains:
        - concept_id: unique SNOMED CT identifier
        - concept_name: fully specified name (preferred term)
        - hierarchy: concept type (e.g., "body structure", "procedure")
        - synonyms: list of acceptable alternative terms
        - parents: list of "is a" parent concepts with id and name
        - children: list of "is a" child concepts with id and name
        - relationships: list of defining relationships (attribute-value pairs)

    This is the file consumed by main.py's FaissTerminologyIndex. The richer
    the data here, the more context the LLM has to disambiguate candidates.
    You could enrich this further by adding grandparent concepts, reference
    set memberships, or other SNOMED CT metadata.
    """
    logger.info("Building enriched JSONL with parents, children, and relationships...")
    concept_ids_in_subset = set(terminology["concept_id"].to_list())

    # Pre-build per-concept relationship lists from the full relationship table,
    # filtered to source concepts in our subset.
    rels_for_source = relationships.filter(
        pl.col("source_id").is_in(concept_ids_in_subset)
    )

    # Also find child concepts: where a concept in our subset is the
    # *destination* of an "is a" relationship, the source is its child.
    rels_for_dest = relationships.filter(
        pl.col("destination_id").is_in(concept_ids_in_subset)
        & (pl.col("type_id") == IS_A_TYPE_ID)
    )

    parents_map: dict[str, list[dict[str, str]]] = {}
    children_map: dict[str, list[dict[str, str]]] = {}
    attrs_map: dict[str, list[dict[str, str]]] = {}

    for source_id, type_id, dest_id in zip(
        rels_for_source["source_id"].to_list(),
        rels_for_source["type_id"].to_list(),
        rels_for_source["destination_id"].to_list(),
    ):
        dest_name = name_lookup.get(dest_id, "")
        if type_id == IS_A_TYPE_ID:
            parents_map.setdefault(source_id, []).append(
                {"concept_id": dest_id, "concept_name": dest_name}
            )
        else:
            type_name = name_lookup.get(type_id, "")
            attrs_map.setdefault(source_id, []).append(
                {
                    "type_id": type_id,
                    "type_name": type_name,
                    "destination_id": dest_id,
                    "destination_name": dest_name,
                }
            )

    # Build children map (reverse of "is a": child --is a--> parent)
    for child_id, parent_id in zip(
        rels_for_dest["source_id"].to_list(),
        rels_for_dest["destination_id"].to_list(),
    ):
        child_name = name_lookup.get(child_id, "")
        children_map.setdefault(parent_id, []).append(
            {"concept_id": child_id, "concept_name": child_name}
        )

    concept_id_list = terminology["concept_id"].to_list()
    concept_name_list = terminology["concept_name"].to_list()
    hierarchy_list = terminology["hierarchy"].to_list()

    logger.info(
        "Mapped {:,} parent links, {:,} child links, {:,} attribute links",
        sum(len(v) for v in parents_map.values()),
        sum(len(v) for v in children_map.values()),
        sum(len(v) for v in attrs_map.values()),
    )

    rows_written = 0
    with output_path.open("w", encoding="utf-8") as f:
        for concept_id, concept_name, hierarchy in tqdm(
            zip(concept_id_list, concept_name_list, hierarchy_list),
            total=len(concept_id_list),
            desc="Writing JSONL",
        ):
            obj = {
                "concept_id": concept_id,
                "concept_name": concept_name,
                "hierarchy": hierarchy,
                "synonyms": synonyms_lookup.get(concept_id, []),
                "parents": parents_map.get(concept_id, []),
                "children": children_map.get(concept_id, []),
                "relationships": attrs_map.get(concept_id, []),
            }
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")
            rows_written += 1

    return rows_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    rf2_zip = Path(args.rf2_zip)
    output_path = Path(args.output)

    if not rf2_zip.exists():
        raise FileNotFoundError(
            f"RF2 ZIP not found: {rf2_zip}\n"
            "Download the SNOMED CT International Edition RF2 release from\n"
            "https://www.nlm.nih.gov/healthit/snomedct/international.html"
        )

    with tempfile.TemporaryDirectory(prefix="snomed-rf2-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        concept_path, description_path, relationship_path = extract_snapshot_files(
            rf2_zip, temp_dir
        )
        concepts = load_active_concepts(concept_path)
        descriptions = load_active_descriptions(description_path)
        relationships = load_active_relationships(relationship_path)
        terminology = build_flat_terminology(concepts, descriptions)
        name_lookup = build_preferred_name_lookup(concepts, descriptions)
        synonyms_lookup = build_synonym_lookup(terminology, descriptions)

    jsonl_rows = build_enriched_jsonl(
        terminology,
        relationships,
        name_lookup,
        synonyms_lookup,
        output_path,
    )
    print(f"JSONL: {output_path}  ({jsonl_rows} concepts)")


if __name__ == "__main__":
    main()
