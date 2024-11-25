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
    chainlit run bin/chat-chainlit.py
    ```
8. Access the app at http://localhost:8000 🎉

### Docker Setup

Follow these

#### Build Docker image

```bash
docker build -t reactome-chatbot .
```

##### Pull embeddings

list embeddings available

```bash
docker run --env-file .env -v $(pwd)/embeddings:/app/embeddings/ reactome-chatbot /bin/bash -c "./bin/embeddings_manager ls-remote"
```

and pull the one you want

```bash
docker run --env-file .env -v $(pwd)/embeddings:/app/embeddings/ reactome-chatbot /bin/bash -c "./bin/embeddings_manager install <the-embedding-from-ls-remote>"
```

The project uses Docker Compose to manage the PostgreSQL database. The configuration for the database is stored in the `docker-compose.yml` file, and the environment variables are stored in the `.env` file.

To start the PostgreSQL database, run the following command:

```bash
docker-compose up -d
```

## Usage

### Getting Embeddings

Please refer to the [Embeddings Manager documentation](docs/embeddings_manager.md) for updated steps for either downloading or generating embeddings.

#### Dependencies

Reactome Dockerized Graph database from DockerHub: [reactome/graphdb](https://hub.docker.com/r/reactome/graphdb)

To generate embeddings using the embedding generator script, use the following command:

```bash
python bin/embeddings_manager.py make openai/text-embedding-3-large/reactome/Release89 --openai-key=<your-key>
```
This command will generate embeddings using the specified OpenAI API key.

To generate embeddings inside docker run:
```bash
mkdir embeddings;
docker run --env-file .env --net=host -v $(pwd)/embeddings/:/apt/embeddings/ --rm reactome-chatbot bash -c "/app/bin/embedding_generator;"
```


## Configuration

### Environment Variables
You can set environment variables for the ChatBot and other components:

- `OPENAI_API_KEY`: API key for the ChatBot (required)

```bash
export OPENAI_API_KEY=your_openai_api_key
```

You can also use a `.env` file to set the environment variable for the chatbot.

## Running the UI

To run the UI, use the following command:

```bash
poetry run chainlit run bin/chat-chainlit.py -w
```
or with docker

```bash
docker run -v $(pwd)/bin:/app/bin -v$(pwd)/src:/app/src reactome-chatbot /bin/bash -c "chainlit run bin/chat-chainlit.py -w"
```
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

To run these inside of docker run a command like
```bash
docker build -t reactome-chatbot .; docker run -v $(pwd)/bin:/app/bin -v$(pwd)/src:/app/src reactome-chatbot /bin/bash -c "poetry run ruff check ."
sudo chown $(id -u):$(id -g) * -R
```

## Contributing
Contributions to the Reactome ChatBot project are welcome! If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

Please make sure to follow our contributing guidelines and code of conduct.

## License

This project is licensed under the [MIT License](LICENSE).
```

Feel free to adjust the instructions and details as needed for your project!
