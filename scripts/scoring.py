from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import typer


def iou_per_class(
    user_annotations: pd.DataFrame, target_annotations: pd.DataFrame
) -> dict:
    """
    Calculate the IoU metric for each class in a set of annotations.

    Returns a dict mapping concept_id to IoU score. Returns an empty dict if both
    inputs are empty.
    """
    if user_annotations.empty and target_annotations.empty:
        return {}

    # Get mapping from note_id to index in array
    docs = np.unique(
        np.concatenate([user_annotations.note_id, target_annotations.note_id])
    )
    doc_index_mapping = dict(zip(docs, range(len(docs))))

    # Identify union of categories in GT and PRED
    cats = np.unique(
        np.concatenate([user_annotations.concept_id, target_annotations.concept_id])
    )

    # Find max character index in GT or PRED
    max_end = np.max(np.concatenate([user_annotations.end, target_annotations.end]))

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

    n_rows = docs.shape[0]
    n_cols = max_end

    def build_class_matrices(annot_df):
        matrices = {}
        for concept_id, group in annot_df.groupby("concept_id"):
            mtx = sp.lil_array((n_rows, n_cols), dtype=bool)
            for row in group.itertuples():
                mtx[doc_index_mapping[row.note_id], row.start : row.end] = True  # noqa: E203
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


def macro_character_iou(predicted: pd.DataFrame, actual: pd.DataFrame) -> float:
    """Macro-averaged character IoU for string span classification."""
    ious = iou_per_class(predicted, actual)
    if not ious:
        return 0.0
    return np.mean(list(ious.values()))


def support_weighted_character_iou(
    predicted: pd.DataFrame, actual: pd.DataFrame
) -> float:
    """Support-weighted character IoU for string span classification.

    Support is the number of span instances (not the number of characters) of each class
    in the evaluation set.
    """
    ious = iou_per_class(predicted, actual)

    # Calculate support (number of GT spans) per class
    cats, counts = np.unique(actual.concept_id, return_counts=True)
    support_mapping = dict(zip(cats, counts))

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
    user_annotations = pd.read_csv(user_annotations_path)
    target_annotations = pd.read_csv(target_annotations_path)
    macro_iou = macro_character_iou(user_annotations, target_annotations)
    weighted_iou = support_weighted_character_iou(user_annotations, target_annotations)
    print(f"macro-averaged character IoU: {macro_iou:0.4f}")
    print(f"support-weighted character IoU: {weighted_iou:0.4f}")


if __name__ == "__main__":
    typer.run(main)
