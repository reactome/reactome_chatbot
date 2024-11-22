# Reactome ChatBot

The Reactome ChatBot is an interactive tool that provides information about biological entities and processes using Advanced RAG techniques. It leverages the Reactome database to retrieve relevant information based on user queries.


## Installation

### Prerequisites

- Python 3.12
- Poetry 1.8 (for dependency management)
- Docker (for running the PostgreSQL database and potentially the application)

### Installation Steps

#### Clone the repository:

```bash
git clone https://github.com/reactome/reactome_chatbot.git
```
#### Navigate to the project directory:

```bash
cd reactome_chatbot
```

#### Install dependencies using Poetry:

```bash
poetry install
```

### Docker Setup


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

This will run the app through bin/chat-fastapi.py where the user will need to fill in an hcaptcha to access the chat interfaces. This should be used in production.

if you want to run chainlit directly run:

```bash
docker run --env-file .env -p 8000:8000 -v $(pwd)/embeddings:/app/embeddings reactome-chatbot /bin/bash -c "chainlit run bin/chat-chainlit.py -w"
```

## Usage

### Running the ChatBot

#### Interactively
To run the ChatBot interactively, execute the following command:

```bash
poetry run bin/chat-repl
```
This will start the ChatBot in interactive mode, allowing users to input queries and receive responses in real-time.

#### Providing Queries Non-interactively
You can also provide queries non-interactively by passing them as command-line arguments.

For example:

```bash
poetry run bin/chat-repl --query "What is TP53 involved in?"
```
This will execute the ChatBot with the provided query and print the response.

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
