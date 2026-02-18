"""Test that key packages can be imported successfully."""

import importlib

import pytest

packages = [
    "accelerate",
    "datasets",
    "einops",
    "faiss",
    "gensim",
    "langchain_community",
    "langchain",
    "loguru",
    "numpy",
    "pandas",
    "peft",
    "ray",
    "scipy",
    "sentence_transformers",
    "sklearn",
    "spacy",
    "timm",
    "torch",
    "transformers",
    "vllm",
]


@pytest.mark.parametrize("package_name", packages, ids=packages)
def test_import(package_name):
    """Test that certain dependencies are importable."""
    importlib.import_module(package_name)


def test_spacy_basic():
    """Test spaCy basic NLP pipeline (CPU only)."""
    import spacy
    from spacy.tokens import DocBin

    nlp = spacy.blank("en")
    training_data = [
        ("Tokyo Tower is 333m tall.", [(0, 11, "BUILDING")]),
    ]

    db = DocBin()
    for text, annotations in training_data:
        doc = nlp(text)
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            ents.append(span)
        doc.ents = ents
        db.add(doc)


def test_faiss_cpu_index():
    """Test FAISS CPU functionality with index creation, add, and search."""
    import faiss
    import numpy as np

    d = 64  # dimension
    nb = 1000  # database size
    nq = 10  # number of queries
    k = 4  # number of nearest neighbors

    np.random.seed(42)
    xb = np.random.random((nb, d)).astype("float32")
    xq = np.random.random((nq, d)).astype("float32")

    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)

    index = faiss.IndexFlatIP(d)
    assert index.is_trained

    index.add(xb)
    assert index.ntotal == nb

    D, I = index.search(xq, k)

    assert D.shape == (nq, k)
    assert I.shape == (nq, k)
    assert np.all(D <= 1.0 + 1e-5)
    assert np.all(D >= -1.0 - 1e-5)
    assert np.all(I >= 0)
    assert np.all(I < nb)
