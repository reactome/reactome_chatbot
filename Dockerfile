# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables for the virtual environment path
ENV VENV_PATH="/app/.venv"

# Set PYTHONPATH environment variable to include the src directory
ENV PYTHONPATH="/app/src:/app/.venv/lib/python3.9/site-packages"
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY pyproject.toml poetry.lock ./

# Install specific versions of filelock, virtualenv, and poetry
RUN pip install filelock==3.15.4 virtualenv==20.26.3 poetry==1.8.3

# Set poetry to create virtual environment in the project directory
RUN poetry config virtualenvs.in-project true

# Install dependencies without dev dependencies
RUN poetry install --no-root --with dev

# Adjust PATH to include the virtual environment's bin directory
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Copy the rest of the application code into the container
COPY . .

# Ensure the virtual environment is activated in the shell
ENV PATH="/app/.venv/bin:$PATH"

# Make all files in the bin directory executable
RUN chmod +x bin/*

CMD ["uvicorn", "bin.chat-fastapi:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "error"]
