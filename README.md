# Reactome ChatBot

The Reactome ChatBot is an interactive tool that provides information about biological entities and processes using Advanced RAG techniques. It leverages the Reactome database to retrieve relevant information based on user queries.


## Installation

### Prerequisites

- **Minimum requirements:**
    + Python 3.12
    + [Poetry](https://python-poetry.org/docs/#installation) `1.8.*`
- **Requirements for running the complete application:**
    + [Docker](https://docs.docker.com/get-started/get-docker/)
    + [Docker Compose](https://docs.docker.com/compose/install/)

### Quick Start

Follow these steps to run the barebones Chainlit application.

1. Clone the repository:
    ```bash
    git clone https://github.com/reactome/reactome_chatbot.git
    ```
2. Navigate to the project directory:
    ```bash
    cd reactome_chatbot
    ```
3. Install dependencies using Poetry:
    ```bash
    poetry install
    ```
4. Add your OpenAI key. The chatbot cannot answer without one:
    ```bash
    cp env_template .env
    # then edit .env and set OPENAI_API_KEY
    ```
5. Put `./src` on the `PYTHONPATH`. The entry points import from there, and
   nothing sets it for you:
    ```bash
    export PYTHONPATH="./src:$PYTHONPATH"
    ```
6. List embeddings available for download. `poetry run` puts the project's
   dependencies on the path:
    ```bash
    poetry run ./bin/embeddings_manager ls-remote
    ```
7. Install your chosen embeddings. These are multi-gigabyte downloads:
    ```bash
    poetry run ./bin/embeddings_manager install openai/text-embedding-3-large/reactome/ReleaseXX
    ```
8. Run the Chainlit application:
    ```bash
    poetry run chainlit run bin/chat-chainlit.py -w
    ```
9. Access the app at http://localhost:8000 🎉

### Docker Setup

The project uses Docker Compose to manage the PostgreSQL database. The configuration for the database is stored in the `docker-compose.yml` file, and the environment variables are stored in the `.env` file.

Follow these steps to run the complete application in Docker.

1. Create a copy of the `env_template` file and name it `.env`:
    ```bash
    cp env_template .env
    ```
2. Configure the application by editing environment variables in `.env`:
    - `OPENAI_API_KEY`: add your OpenAI key.
    - `CLOUDFLARE_SECRET_KEY`: keep blank to disable captcha.
    - `CHAINLIT_IMAGE=reactome-chatbot`: set this to use your local docker build.
    - Use the following variables to configure Auth0:
        + This will enable Chainlit user-login and chat history.
        ```
        OAUTH_AUTH0_CLIENT_ID
        OAUTH_AUTH0_CLIENT_SECRET
        OAUTH_AUTH0_DOMAIN
        ```
3. List embeddings available for download:
    ```bash
    docker compose run --rm chainlit /bin/bash -c "./bin/embeddings_manager ls-remote"
    ```
4. Install your chosen embeddings:
    ```bash
    docker compose run --rm chainlit /bin/bash -c "./bin/embeddings_manager install openai/text-embedding-3-large/reactome/ReleaseXX"
    ```
5. Build the Docker image (do this every time you make local changes):
    ```bash
    docker build -t reactome-chatbot .
    ```
6. Start the Chainlit application and PostgrSQL database in Docker containers:
    ```bash
    docker-compose up

    # To run it in the background, use:
    # docker-compose up -d
    ```
7. Access the app at http://localhost:8000 🎉


## Embeddings & Documents Bundles

The ChatBot's knowledge of a given data source is generated using the latest data release, resulting in a bundle of embedded information and/or text documents. For simplicity, we refer to these bundles as **Embeddings** throughout this document.

In the case of Reactome, embeddings bundles are generated once per release from [reactome/graphdb](https://hub.docker.com/r/reactome/graphdb) releases from DockerHub and uploaded to AWS S3 for easy retrieval.

User guide embeddings are generated separately from Reactome website documentation and use a date-based version identifier (for example, `userguide/2025-06`). See [Embeddings Manager documentation](docs/embeddings_manager.md) for details.

### Embeddings Manager Script

All aspects of generating, managing, uploading, and retrieving embeddings bundles are handled by the `./bin/embeddings_manager` script.
- Basic usage is covered in the **_Quick Start_** guide above.
- See the [Embeddings Manager documentation](docs/embeddings_manager.md) for more information.


## Developers

### Code Quality

All tool configuration lives in `pyproject.toml`. Ruff handles linting, import
sorting, and formatting (it replaces `black` and `isort`).

```bash
poetry run ruff check .          # lint (add --fix to autofix)
poetry run ruff format .         # format
poetry run mypy                  # type check
poetry run pytest                # tests
```

CI runs all four on every pull request and on pushes to `main`.

Optionally, run the same checks on every commit:

```bash
pipx install pre-commit && pre-commit install
```

### Tests

```bash
poetry run pytest
poetry run pytest -m "not requires_retrieval_stack"   # no ML deps needed
```

Tests that need an installed embeddings bundle are marked `requires_embeddings`
and skip themselves when none is present. See `tests/README.md` for what is
covered and why coverage is currently narrow.


### Contributing
Contributions to the Reactome ChatBot project are welcome! If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

Please make sure to follow our contributing guidelines and code of conduct.

## License

This project is licensed under the [MIT License](LICENSE).
