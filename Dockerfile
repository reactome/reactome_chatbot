# Use official python-slim image
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Create non-root user and group. The image used to run everything as root; a
# compromise in chainlit or any dependency then owned the container.
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
    rm -rf $POETRY_VENV

# NLTK data, at build time. BM25 tokenises with
# word_tokenize(..., language="english"), which needs punkt_tab -- without it
# every retrieval either downloads it on first use or fails. It lands in
# appuser's home because the download runs as appuser.
RUN /app/.venv/bin/python -m nltk.downloader punkt_tab

# Copy essential application files.
#
# `--chmod` applies to the directories it creates as well as the files in them,
# and a directory without the execute bit cannot be traversed -- not even by
# its owner. `--chmod=400` on bin/ and public/ therefore produced an image
# whose own entrypoint could not be read:
#
#     ERROR: Error loading ASGI app. Could not import module "bin.chat-fastapi".
#
# with /app/bin at `dr--------`. It built and pushed cleanly, because nothing
# ran it.
#
# 500 rather than 400 for these two, so their directories can be entered. That
# also marks the files inside executable, which is what bin/ wants anyway and
# is harmless for the static assets in public/. Everything else keeps the mode
# it had.
COPY --chown=appuser --chmod=700 .chainlit/ /app/.chainlit/
COPY --chown=appuser --chmod=500 bin/ /app/bin/
COPY --chown=appuser --chmod=500 public/ /app/public/
COPY --chown=appuser --chmod=700 src/ /app/src/
COPY --chown=appuser --chmod=400 chainlit.md /app/
COPY --chown=appuser --chmod=400 config_default.yml /app/
COPY --chown=appuser --chmod=400 LICENSE /app/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Fail the build rather than publish an image whose own entrypoint is
# unreadable. This runs as appuser, which is who has to read these at runtime.
RUN test -r /app/bin/chat-fastapi.py \
 && test -r /app/bin/chat-chainlit.py \
 && test -r /app/config_default.yml \
 && test -r /app/chainlit.md \
 && python -c "import sys; sys.path.insert(0, '/app/src'); import agent.graph"

CMD ["uvicorn", "bin.chat-fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
