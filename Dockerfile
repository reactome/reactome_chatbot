# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for the virtual environment path
ENV VENV_PATH="/app/.venv"

# Set PYTHONPATH environment variable to include the src directory
ENV PYTHONPATH="/app/src"
# Install system dependencies
# libpq5 is for python package psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY pyproject.toml poetry.lock ./

# Install specific versions of filelock, virtualenv, and poetry
RUN pip install filelock==3.15.4 virtualenv==20.26.6 poetry==1.8.4

# Set poetry to create virtual environment in the project directory
RUN poetry config virtualenvs.in-project true

# Install dependencies without dev dependencies
RUN poetry install --no-root --without dev

# Download NLTK data
RUN poetry run python -m nltk.downloader punkt_tab

# Adjust PATH to include the virtual environment's bin directory
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Copy the rest of the application code into the container
COPY . .

# Ensure the virtual environment is activated in the shell
ENV PATH="/app/.venv/bin:$PATH"

# Make all files in the bin directory executable
RUN chmod +x bin/*

CMD ["uvicorn", "bin.chat-fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
