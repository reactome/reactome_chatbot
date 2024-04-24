# Reactome ChatBot

The Reactome ChatBot is an interactive tool that provides information about biological entities and processes using natural language processing techniques. It leverages the Reactome database to retrieve relevant information based on user queries.


## Installation

### Prerequisites

- Python 3.x
- Poetry (for dependency management)

### Installation Steps

#### Clone the repository:

```bash
   git clone https://github.com/your_username/reactome_chatbot.git
```
#### Navigate to the project directory:

```bash
cd reactome_chatbot
```

#### Install dependencies using Poetry:

```bash
poetry install
```
## Usage

usage: chat.py [-h] [--openai-key OPENAI_KEY] [--query QUERY] [--verbose]

Reactome ChatBot

optional arguments:
  -h, --help            show this help message and exit
  --openai-key OPENAI_KEY
                        API key for OpenAI
  --query QUERY         Query string to run
  --verbose             Enable verbose mode

### Generating Embeddings

#### Dependencies

Reactome Dockerized Graph database form DockerHub: https://hub.docker.com/r/reactome/graphdb

To generate embeddings using the embedding generator script, use the following command:

```bash
poetry run python bin/embedding_generator.py --openai-key=<your-key>
```
This command will generate embeddings using the specified OpenAI API key.


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

## Configuration

### Environment Variables
You can set environment variables for the ChatBot and other components:

OPENAI_API_KEY: API key for the ChatBot (required)

```bash
export OPENAI_API_KEY=your_openai_api_key
```

You can also use a .env file to set the environment variable for the chatbot.

## To Run the UI run the following command

```bash
poetry run chainlit run bin/app.py -w
```

## Contributing
Contributions to the Reactome ChatBot project are welcome! If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
