# ==========================================
# 1. Base Image (PyTorch with CUDA support)
# ==========================================
# FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime  # Use this for GPU

# CPU-only base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# ==========================================
# 2. Environment Setup
# ==========================================
# Prevent Python from buffering stdout (so you see logs instantly)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# ==========================================
# 3. System Dependencies
# ==========================================
# Install UV and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential git curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ==========================================
# 4. Python Dependencies (The "UV" way)
# ==========================================
# Copy ONLY the dependency files first for caching
COPY pyproject.toml uv.lock ./

# Install dependencies from the lockfile
# --locked: fails if uv.lock doesn't match pyproject.toml
# --no-dev: skips development tools (pytest, ruff, etc.)
# --no-install-project: installs libraries (torch, numpy) but not your code yet
RUN uv sync --locked --no-dev --no-install-project

# ==========================================
# 5. Application Code
# ==========================================
# Copy the source code, data DVC file and the README
COPY src/ ./src/
COPY data.zip.dvc ./
COPY .dvc/ ./.dvc/
COPY tasks.py ./
COPY configs/ ./configs/
COPY README.md ./
COPY train.sh ./

# Install the project itself (so imports like 'import mlopsproject' work)
RUN uv sync --locked --no-dev

# Make train.sh executable
RUN chmod +x train.sh

# ==========================================
# 6. Runtime
# ==========================================
# Run the training script
ENTRYPOINT ["./train.sh"]
