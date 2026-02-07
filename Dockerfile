# ---------------------------------------------------------------------------
# Multi-stage Dockerfile for skewed-sequences
# ---------------------------------------------------------------------------
# Build:   docker build -t skewed-sequences .
# Run:     docker run --rm skewed-sequences skseq --help
# Example: docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/mlruns:/app/mlruns \
#            skewed-sequences skseq train --loss-type mse
# ---------------------------------------------------------------------------

# ---- Stage 1: build wheel with Poetry ------------------------------------
FROM python:3.11-slim AS builder

ENV POETRY_VERSION=2.2.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /build
COPY pyproject.toml poetry.lock* ./
COPY skewed_sequences ./skewed_sequences
COPY README.md ./

# Install only main dependencies (no dev/docs groups)
RUN poetry install --only main --no-root \
    && poetry build -f wheel

# ---- Stage 2: lean runtime image -----------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Mikhail Gritskikh" \
      description="Skewed sequences — loss function analysis for transformer NNs"

WORKDIR /app

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the built wheel and install it (gives us the `skseq` entry point)
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

# Copy project files needed at runtime (data dirs, configs, mlruns mount point)
COPY skewed_sequences ./skewed_sequences
COPY data ./data
COPY reports ./reports

# Create volume mount-points so results persist outside the container
VOLUME ["/app/data", "/app/mlruns", "/app/models", "/app/reports"]

# Default: show help
ENTRYPOINT ["skseq"]
CMD ["--help"]
