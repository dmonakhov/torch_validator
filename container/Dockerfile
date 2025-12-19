FROM nvcr.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Install Python, uv, and build tools (gcc + headers required by Triton)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    python3 python3-dev python3-venv curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# Create install directory
WORKDIR /opt/torch_validator

# Copy source
COPY . .

# Create venv and install torch + torch_validator
RUN uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && echo "current git head: $(git rev-parse HEAD)" \
    && uv pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install -e .

# Set environment
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV TORCHINDUCTOR_COMPILE_THREADS=1
ENV OMP_NUM_THREADS=8

WORKDIR /opt/torch_validator
