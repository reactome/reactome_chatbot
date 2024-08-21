# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

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
RUN poetry install --no-root --no-dev

# Copy the rest of the application code into the container
COPY . .

# Ensure the virtual environment is activated in the shell
ENV PATH="/app/.venv/bin:$PATH"

# Specify the command to run on container start
CMD ["chainlit", "run", "bin/app.py", "-w"]
