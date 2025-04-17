# Use official python-slim image
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Create non-root user and group
RUN \
    groupadd -g 1000 appgroup && \
    useradd -m -u 3001 -g appgroup appuser && \
    chown -R appuser /app && \
    chmod -R 700 /app
USER appuser:appgroup

# Install Python dependencies
COPY --chown=appuser --chmod=400 poetry.lock /app/
COPY --chown=appuser --chmod=700 pyproject.toml /app/
ARG POETRY_VERSION=1.8.4
ENV POETRY_VENV="/home/appuser/.poetry"
RUN \
    python -m venv $POETRY_VENV && \
    $POETRY_VENV/bin/pip install -U pip setuptools && \
    $POETRY_VENV/bin/pip install poetry~=$POETRY_VERSION && \
    $POETRY_VENV/bin/poetry config virtualenvs.in-project true && \
    $POETRY_VENV/bin/poetry install --no-root --without dev && \
    rm -rf $POETRY_VENV && \
    . /app/.venv/bin/activate

# Copy essential application files
COPY --chown=appuser --chmod=700 .chainlit/ /app/.chainlit/
COPY --chown=appuser --chmod=400 bin/ /app/bin/
COPY --chown=appuser --chmod=400 public/ /app/public/
COPY --chown=appuser --chmod=700 src/ /app/src/
COPY --chown=appuser --chmod=400 chainlit.md /app/
COPY --chown=appuser --chmod=400 config_default.yml /app/
COPY --chown=appuser --chmod=400 LICENSE /app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

CMD ["uvicorn", "bin.chat-fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
