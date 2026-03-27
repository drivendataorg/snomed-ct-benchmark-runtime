# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars>=1",
#     "typer",
# ]
# ///
from pathlib import Path

import polars as pl
import typer


def main(train_notes_path: Path, train_annotations_path: Path):
    # Read training data
    train_notes = pl.read_csv(train_notes_path)
    train_annotations = pl.read_csv(train_annotations_path)

    # Subset to smoke test notes
    smoke_test_note_ids = [
        "10302979-DS-5",
        "10097089-DS-8",
        "10124346-DS-4",
        "10043750-DS-6",
        "10060142-DS-9",
    ]
    smoke_notes = train_notes.filter(pl.col("note_id").is_in(smoke_test_note_ids))
    smoke_annotations = train_annotations.filter(
        pl.col("note_id").is_in(smoke_test_note_ids)
    )

    # Cast to expected dtypes and write to data/
    smoke_notes = smoke_notes.cast({"note_id": pl.String})
    smoke_annotations = smoke_annotations.cast({
        "note_id": pl.String,
        "start": pl.Int64,
        "end": pl.Int64,
        "concept_id": pl.String,
    })
    smoke_notes.write_csv("data/test_notes.csv")
    smoke_annotations.write_csv("data/smoke_test_annotations.csv")


if __name__ == "__main__":
    typer.run(main)
