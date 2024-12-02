# Reactome ChatBot

The Reactome ChatBot is an interactive tool that provides information about biological entities and processes using Advanced RAG techniques. It leverages the Reactome database to retrieve relevant information based on user queries.


## Installation

### Prerequisites

- Python 3.12
- Poetry 1.8 (for dependency management)
- _Optional_: Docker (for running the complete application with PostgreSQL database)

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
4. Verify your `PYTHONPATH` environment variable includes `./src`:
    ```bash
    echo $PYTHONPATH
    # ./src
    ```
5. List embeddings available for download:
    ```bash
    ./bin/embeddings_manager ls-remote
    ```
6. Install your chosen embeddings:
    ```bash
    ./bin/embeddings_manager install openai/text-embedding-3-large/reactome/ReleaseXX
    ```
7. Run the Chainlit application:
    ```
    chainlit run bin/chat-chainlit.py -w
    ```
8. Access the app at http://localhost:8000 🎉

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
    docker run --env-file .env -v $(pwd)/embeddings:/app/embeddings/ reactome-chatbot /bin/bash -c "./bin/embeddings_manager ls-remote"
    ```
4. Install your chosen embeddings:
    ```bash
    docker run --env-file .env -v $(pwd)/embeddings:/app/embeddings/ reactome-chatbot /bin/bash -c "./bin/embeddings_manager install openai/text-embedding-3-large/reactome/ReleaseXX"
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

The ChatBot's knowledge of a given data source is generated directly using the latest data release, resulting in a bundle of embedded information and/or text documents. For simplicity, we refer to these bundles as **Embeddings** throughout this document.

In the case of Reactome, embeddings bundles are generated once per release from [reactome/graphdb](https://hub.docker.com/r/reactome/graphdb) releases from DockerHub and uploaded to AWS S3 for easy retrieval.

### Embeddings Manager Script

All aspects of generating, managing, uploading, and retrieving embeddings bundles are handled by the `./bin/embeddings_manager` script.
- Basic usage is covered in the **_Quick Start_** guide above.
- See the [Embeddings Manager documentation](docs/embeddings_manager.md).


## Code Quality

To do main consistency checks
```bash
poetry run ruff check .
```

To make style consistent

```bash
poetry run black .
```

To make sure imports are organized


```bash
poetry run isort .
```


## Contributing
Contributions to the Reactome ChatBot project are welcome! If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

Please make sure to follow our contributing guidelines and code of conduct.

## License

This project is licensed under the [MIT License](LICENSE).
```

Feel free to adjust the instructions and details as needed for your project!
