# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Set PYTHONPATH environment variable to include the src directory
#ENV PYTHONPATH="/app/src:/app:$PYTHONPATH"
ENV PYTHONPATH="/app/src:/app/.venv/lib/python3.9/site-packages:$PYTHONPATH"
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY pyproject.toml poetry.lock ./

# Install specific versions of filelock, virtualenv, and poetry
RUN pip install filelock==3.0.0 virtualenv==20.8.0 poetry

# Set poetry to create virtual environment in the project directory
RUN poetry config virtualenvs.in-project true

# Install dependencies without dev dependencies
RUN poetry install --no-root --no-dev

# Copy the rest of the application code into the container
COPY . .

# Ensure the virtual environment is activated in the shell
ENV PATH="/app/.venv/bin:$PATH"

# Specify the command to run on container start
CMD ["chainlit", "run", "bin/app.py", "-w"]
