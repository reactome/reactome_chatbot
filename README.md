# Reactome ChatBot

The Reactome ChatBot is an interactive tool that provides information about biological entities and processes using Advanced RAG techniques. It leverages the Reactome database to retrieve relevant information based on user queries.


## Installation

### Prerequisites

- Python 3.x
- Poetry (for dependency management)
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

The project uses Docker Compose to manage the PostgreSQL database. The configuration for the database is stored in the `docker-compose.yml` file, and the environment variables are stored in the `.env` file.

To start the PostgreSQL database, run the following command:

```bash
docker-compose up -d
```

## Usage

### Running the ChatBot

#### Interactively
To run the ChatBot interactively, execute the following command:

```bash
poetry run python bin/chat.py
```
This will start the ChatBot in interactive mode, allowing users to input queries and receive responses in real-time.

#### Providing Queries Non-interactively
You can also provide queries non-interactively by passing them as command-line arguments.

For example:

```bash
poetry run python bin/chat.py --query "What is TP53 involved in?"
```
This will execute the ChatBot with the provided query and print the response.

### Generating Embeddings

#### Dependencies

Reactome Dockerized Graph database from DockerHub: [reactome/graphdb](https://hub.docker.com/r/reactome/graphdb)

To generate embeddings using the embedding generator script, use the following command:

```bash
poetry run python bin/embedding_generator.py --openai-key=<your-key>
```
This command will generate embeddings using the specified OpenAI API key.

To generate embeddings inside docker run:
```bash
mkdir embeddings;
docker run --net=host -v $(pwd)/embeddings/:/apt/embeddings/ --rm reactome-chatbot bash -c "python /app/bin/embedding_generator.py --openai-key=TOKEN;
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
poetry run chainlit run bin/app.py -w
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
poetry run iosort . 
```

## Contributing
Contributions to the Reactome ChatBot project are welcome! If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

Please make sure to follow our contributing guidelines and code of conduct.

## License

This project is licensed under the [MIT License](LICENSE).
```

Feel free to adjust the instructions and details as needed for your project!
