FROM python:3.12-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements_api.txt requirements_api.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install --no-cache-dir --upgrade pip

# Install CPU-only PyTorch wheels
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install API runtime deps
RUN pip install --no-cache-dir -r requirements_api.txt

# Install code 
RUN pip install --no-cache-dir --verbose . --no-deps


ENTRYPOINT ["sh", "-c", "uvicorn src.mlopsproject.api:app --host 0.0.0.0 --port ${PORT:-8080}"]