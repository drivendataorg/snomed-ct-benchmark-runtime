"""Test GPU functionality of key libraries.

All tests in this file are skipped if no GPU is detected.
"""

import os
import subprocess

import numpy as np
import pytest


def is_gpu_available():
    """Check if nvidia-smi is available, indicating GPU presence."""
    try:
        return subprocess.check_call(["nvidia-smi"]) == 0
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
        return False


GPU_AVAILABLE = is_gpu_available()
pytestmark = pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")


def test_torch_gpu_matmul():
    """Test that PyTorch can perform matrix multiplication on GPU and get correct results."""
    import torch

    assert torch.cuda.is_available(), "CUDA should be available"

    # Create known matrices on GPU
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")

    result = torch.matmul(a, b)
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], device="cuda")

    assert result.device.type == "cuda"
    torch.testing.assert_close(result, expected)


def test_cupy_gpu_computation():
    """Test that CuPy can perform computation on GPU and get correct results."""
    import cupy as cp

    a = cp.array([[1.0, 2.0], [3.0, 4.0]])
    b = cp.array([[5.0, 6.0], [7.0, 8.0]])

    result = cp.dot(a, b)
    expected = cp.array([[19.0, 22.0], [43.0, 50.0]])

    assert result.device.id >= 0
    cp.testing.assert_array_almost_equal(result, expected)


def test_spacy_gpu():
    """Test spaCy can activate GPU and run a pipeline on it."""
    import spacy

    spacy.require_gpu()
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    doc = nlp("This is sentence one. This is sentence two.")
    sentences = list(doc.sents)
    assert len(sentences) == 2


def test_faiss_gpu_index():
    """Test FAISS GPU functionality with index creation, transfer, add, and search."""
    import faiss

    d = 64  # dimension
    nb = 10000  # database size
    nq = 100  # number of queries
    k = 10  # number of nearest neighbors

    np.random.seed(42)
    xb = np.random.random((nb, d)).astype("float32")
    xq = np.random.random((nq, d)).astype("float32")

    # Create CPU index first
    cpu_index = faiss.IndexFlatL2(d)

    # Transfer to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    # Add vectors on GPU
    gpu_index.add(xb)
    assert gpu_index.ntotal == nb

    # Search on GPU
    D_gpu, I_gpu = gpu_index.search(xq, k)

    assert D_gpu.shape == (nq, k)
    assert I_gpu.shape == (nq, k)
    assert np.all(D_gpu >= 0)
    assert np.all(I_gpu >= 0)
    assert np.all(I_gpu < nb)

    # Compare with CPU results for correctness
    cpu_index.add(xb)
    D_cpu, I_cpu = cpu_index.search(xq, k)

    # GPU and CPU should return the same nearest neighbors
    np.testing.assert_array_equal(I_gpu, I_cpu)
    np.testing.assert_allclose(D_gpu, D_cpu, rtol=1e-5)


def test_vllm_generate():
    """Test that vLLM can load a tiny model on GPU and generate tokens."""
    from vllm import LLM, SamplingParams

    model_path = os.environ.get("VLLM_TEST_MODEL_PATH", "facebook/opt-125m")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.3,
        max_model_len=64,
        enforce_eager=True,
    )
    params = SamplingParams(max_tokens=10, temperature=0.0)
    outputs = llm.generate(["Hello, world"], params)

    assert len(outputs) == 1
    assert len(outputs[0].outputs) == 1
    assert len(outputs[0].outputs[0].text) > 0
