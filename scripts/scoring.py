# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2",
#     "polars>=1",
#     "scipy>=1",
#     "typer",
# ]
# ///
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import typer

COLUMNS = ["note_id", "start", "end", "concept_id"]
DTYPES = {
    "note_id": pl.String,
    "start": pl.Int64,
    "end": pl.Int64,
    "concept_id": pl.String,
}


def iou_per_class(
    user_annotations: pl.DataFrame, target_annotations: pl.DataFrame
) -> dict:
    """
    Calculate the IoU metric for each class in a set of annotations.

    Returns a dict mapping concept_id to IoU score. Returns an empty dict if both
    inputs are empty.
    """
    if user_annotations.is_empty() and target_annotations.is_empty():
        return {}

    # Get mapping from note_id to index in array
    all_note_ids = pl.concat(
        [user_annotations.select("note_id"), target_annotations.select("note_id")]
    )["note_id"]
    docs = all_note_ids.unique().sort()
    doc_index_mapping = {doc: i for i, doc in enumerate(docs.to_list())}

    # Identify union of categories in GT and PRED
    all_concept_ids = pl.concat(
        [
            user_annotations.select("concept_id"),
            target_annotations.select("concept_id"),
        ]
    )["concept_id"]
    cats = all_concept_ids.unique().sort().to_list()

    # Find max character index in GT or PRED
    max_end = max(
        user_annotations["end"].max() or 0,
        target_annotations["end"].max() or 0,
    )

    # Populate per-class boolean matrices for keeping track of character categorization.
    # A separate boolean matrix per class supports overlapping predictions (a character
    # can belong to multiple classes simultaneously).
    # Matrix dimensions are n_docs x n_chars, with True where a character belongs to that class.
    # e.g. for class 1 and class 2 (overlapping spans allowed):
    # gt@1    = [[1 1 1 0 0 0 0 0 0], # gt@2    = [[0 0 0 0 0 0 1 1 1],
    #            [0 0 0 0 1 1 1 1 1]] #            [1 1 0 0 0 0 0 0 0]]
    # pred@1  = [[1 1 0 0 0 0 0 0 0], # pred@2  = [[0 0 0 0 0 0 0 1 1],
    #            [0 0 0 1 1 1 1 1 1]] #            [1 1 1 0 0 0 0 0 0]]
    #
    # itsct@1 = [[1 1 0 0 0 0 0 0 0], # itsct@2 = [[0 0 0 0 0 0 0 1 1],
    #            [0 0 0 0 1 1 1 1 1]] #            [1 1 0 0 0 0 0 0 0]]
    # union@1 = [[1 1 1 0 0 0 0 0 0], # union@2 = [[0 0 0 0 0 0 1 1 1],
    #            [0 0 0 1 1 1 1 1 1]] #            [1 1 1 0 0 0 0 0 0]]
    # IoU@1 = 7 / 9                   # IoU@2 = 4 / 6

    n_rows = docs.len()
    n_cols = max_end

    def build_class_matrices(annot_df: pl.DataFrame) -> dict:
        matrices = {}
        for concept_id, group in annot_df.group_by("concept_id"):
            concept_id = concept_id[0]
            mtx = sp.lil_array((n_rows, n_cols), dtype=bool)
            for row in group.iter_rows(named=True):
                mtx[doc_index_mapping[row["note_id"]], row["start"] : row["end"]] = (
                    True
                )
            matrices[concept_id] = mtx.tocsr()
        return matrices

    gt_matrices = build_class_matrices(target_annotations)
    pred_matrices = build_class_matrices(user_annotations)

    empty = sp.csr_array((n_rows, n_cols), dtype=bool)

    # Calculate IoU per category
    ious = {}
    for cat in cats:
        gt_cat = gt_matrices.get(cat, empty)
        pred_cat = pred_matrices.get(cat, empty)
        intersection = gt_cat.multiply(pred_cat)
        union = (gt_cat + pred_cat).astype(bool)
        ious[cat] = intersection.sum() / union.sum()

    return ious


def macro_character_iou(predicted: pl.DataFrame, actual: pl.DataFrame) -> float:
    """Macro-averaged character IoU for string span classification."""
    ious = iou_per_class(predicted, actual)
    if not ious:
        return 0.0
    return np.mean(list(ious.values()))


def support_weighted_character_iou(
    predicted: pl.DataFrame, actual: pl.DataFrame
) -> float:
    """Support-weighted character IoU for string span classification.

    Support is the number of span instances (not the number of characters) of each class
    in the evaluation set.
    """
    ious = iou_per_class(predicted, actual)

    # Calculate support (number of GT spans) per class
    support_df = actual.group_by("concept_id").len()
    support_mapping = dict(
        zip(
            support_df["concept_id"].to_list(),
            support_df["len"].to_list(),
        )
    )

    total_support = 0
    weighted_sum = 0.0
    for cat, iou in ious.items():
        support = support_mapping.get(cat, 0)
        weighted_sum += iou * support
        total_support += support

    if total_support == 0:
        return 0.0
    return weighted_sum / total_support


def main(
    user_annotations_path: Path,
    target_annotations_path: Path,
):
    """
    Calculate the macro-averaged character IoU metric for each class in a set of annotations.
    """
    user_annotations = pl.read_csv(
        user_annotations_path, schema_overrides=DTYPES
    ).select(COLUMNS)
    target_annotations = pl.read_csv(
        target_annotations_path, schema_overrides=DTYPES
    ).select(COLUMNS)
    macro_iou = macro_character_iou(user_annotations, target_annotations)
    weighted_iou = support_weighted_character_iou(user_annotations, target_annotations)
    print(f"macro-averaged character IoU: {macro_iou:0.4f}")
    print(f"support-weighted character IoU: {weighted_iou:0.4f}")


if __name__ == "__main__":
    typer.run(main)
