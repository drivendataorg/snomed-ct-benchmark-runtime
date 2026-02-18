"""Test GPU functionality of key libraries.

All tests in this file are skipped if no GPU is detected.
"""

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


def test_torch_cuda_available():
    """Test that PyTorch can see CUDA."""
    import torch

    assert torch.cuda.is_available(), "CUDA should be available"


def test_torch_allocate_tensor():
    """Test that PyTorch can allocate a tensor on GPU."""
    import torch

    tensor = torch.zeros(1).cuda()
    assert tensor.device.type == "cuda"


def test_cupy_allocate_array():
    """Test that CuPy can allocate an array on GPU."""
    import cupy as cp

    arr = cp.array([1, 2, 3, 4, 5, 6])
    assert arr.device.id >= 0


def test_spacy_gpu():
    """Test spaCy can use GPU."""
    import spacy

    spacy.require_gpu()


def test_faiss_gpu_available():
    """Test that FAISS can see GPUs."""
    import faiss

    ngpus = faiss.get_num_gpus()
    assert ngpus > 0, "FAISS should detect at least one GPU"


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


def test_vllm_imports():
    """Test that vLLM core classes can be imported (requires GPU)."""
    from vllm import LLM, SamplingParams

    assert LLM is not None
    assert SamplingParams is not None
