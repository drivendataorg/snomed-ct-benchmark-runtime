# SNOMED-CT Benchmark Runtime
# Requires faiss-builder image. Either:
#   - Pull from ACR: docker pull snomedctbenchmarkprodacr.azurecr.io/faiss-builder:v1.13.2
#   - Build locally: docker build -f Dockerfile.faiss -t faiss-builder:v1.13.2 .
#
# Usage:
#   docker build runtime --target runtime \
#     --build-context faiss-builder=docker-image://faiss-builder:v1.13.2 \
#     -t snomed-ct:latest

# Base image with uv and all Python dependencies (except faiss)
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS base
COPY --from=ghcr.io/astral-sh/uv:0.9.25 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    SHELL=/bin/bash

# Create runtime user
ENV RUNTIME_USER=runtimeuser
ENV RUNTIME_UID=1000
ENV RUNTIME_GID=1000

RUN echo "Creating ${RUNTIME_USER} user..." \
    && groupadd --gid ${RUNTIME_GID} ${RUNTIME_USER} \
    && useradd --create-home --gid ${RUNTIME_GID} --no-log-init --uid ${RUNTIME_UID} ${RUNTIME_USER}

# Install system dependencies (no Python - managed by uv)
RUN apt-get update --fix-missing \
    && apt-get install -y \
        libopenblas0 \
        libgomp1 \
        build-essential \
        cmake \
        g++ \
        gcc \
        git \
        libxml2 \
        tzdata \
        wget \
        zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up code execution working directory
RUN mkdir /code_execution && \
    chown -R ${RUNTIME_USER}:${RUNTIME_USER} /code_execution
WORKDIR /code_execution

# Switch to runtime user for Python package installation
USER ${RUNTIME_USER}

# Copy lockfile, pyproject.toml, and .python-version first
COPY --chown=${RUNTIME_USER}:${RUNTIME_USER} uv.lock pyproject.toml .python-version ./

# uv environment variables
ENV UV_FROZEN=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Set up virtual environment path
ENV VIRTUAL_ENV=/code_execution/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python and dependencies (no dev/test deps)
ENV UV_CACHE_DIR=/home/runtimeuser/.cache/uv
ENV UV_PYTHON_CACHE_DIR=/home/runtimeuser/.cache/uv/python
RUN --mount=type=cache,id=uv-cache,target=/home/runtimeuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --no-dev

# Install faiss-gpu from builder image
# Requires: --build-context faiss-builder=docker-image://<faiss-image>
USER root
COPY --from=faiss-builder /build/faiss/build/faiss/python/dist/*.whl /tmp/
COPY --from=faiss-builder /usr/local/lib/libfaiss* /usr/local/lib/
COPY --from=faiss-builder /build/faiss/build/faiss/python/libfaiss_python_callbacks.so /usr/local/lib/
RUN ldconfig

ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

USER ${RUNTIME_USER}
RUN uv pip install --no-cache-dir /tmp/*.whl

USER root
RUN rm /tmp/*.whl

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
USER ${RUNTIME_USER}

# Runtime image
FROM base AS runtime
ENV UV_NO_SYNC=1
CMD ["bash", "/entrypoint.sh"]

# Test image
FROM base AS test-runtime

# Install test dependencies (--inexact to avoid removing faiss)
RUN --mount=type=cache,id=uv-cache,target=/home/runtimeuser/.cache/uv,uid=1000,gid=1000 \
    uv sync --extra test --inexact

# Download a tiny model for vLLM GPU test (network unavailable at runtime)
ENV VLLM_TEST_MODEL_PATH=/code_execution/.test-models/opt-125m
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/opt-125m', local_dir='${VLLM_TEST_MODEL_PATH}')"

USER root
COPY --chown=${RUNTIME_USER}:${RUNTIME_USER} tests ./tests
USER ${RUNTIME_USER}

ENV UV_NO_SYNC=1
CMD ["python", "-m", "pytest", "tests", "-v"]
