# Reactome ChatBot

The Reactome ChatBot is an interactive tool that provides information about biological entities and processes using Advanced RAG techniques. It leverages the Reactome database to retrieve relevant information based on user queries.


## Installation

### Supported Operating Systems
- 64-bit Unix Systems
  - **Linux** (x64)
  - **macOS** (Intel x64; Apple Silicon requires emulations)
  
- **Windows** (not supported natively)
  - Windows 10 version 2004 or higher, or Windows 11
  - Windows users must use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) (Windows Subsystem for Linux).
  - **WSL version 2.1.5 or later** is required. Check your version with:
    ```bash
    wsl --version
    ```
    If outdated, update with:
    ```bash
    wsl --update
    ```
  - Recommended distribution: Ubuntu

### Prerequisites

- **Minimum requirements:**
    + Python 3.12
    + [Poetry](https://python-poetry.org/docs/#installation) `1.8.*`
- **Requirements for running the complete application:**
    + [Docker](https://docs.docker.com/get-started/get-docker/)
    + [Docker Compose](https://docs.docker.com/compose/install/)
    + [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) for Windows users with WSL2 backend enabled (Settings → Resources → WSL integration)

### Option A: Docker Setup (recommended)

The project uses Docker Compose to manage the PostgreSQL database. The configuration for the database is stored in the `docker-compose.yml` file, and the environment variables are stored in the `.env` file.

Follow these steps to run the complete application in Docker.

1. Clone the repository:
    ```bash
    git clone https://github.com/reactome/reactome_chatbot.git
    ```
2. Navigate to the project directory:
    ```bash
    cd reactome_chatbot
    ```
3. Create a copy of the `env_template` file and name it `.env`:
    ```bash
    cp env_template .env
    ```
4. Configure the application by editing environment variables in `.env`:
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
5. List embeddings available for download:
    ```bash
    docker compose run --rm chainlit /bin/bash -c "./bin/embeddings_manager ls-remote"
    ```
6. Install your chosen embeddings (replace `ReleaseXX` with any valid release, e.g. Release89):
    ```bash
    docker compose run --rm chainlit /bin/bash -c "./bin/embeddings_manager install openai/text-embedding-3-large/reactome/ReleaseXX"
    ```
7. Build the Docker image (do this every time you make local changes):
    ```bash
    docker build -t reactome-chatbot .
    ```
8. Start the Chainlit application and PostgrSQL database in Docker containers:
    ```bash
    docker compose up

    # To run it in the background, use:
    # docker compose up -d
    ```
9. Access the app at http://localhost:8000 🎉

### Using Prebuilt Docker image (faster)

Clone and navigate to the project repository. Then in your `.env` file set the required keys as in normal docker setup and set:
```
CHAINLIT_IMAGE=public.ecr.aws/reactome/reactome-chatbot:2044791
```
The part after `:` can be set to any commit ID from the `main` branch.

Then run:
```bash
docker compose up

# Now you don't need to build the image locally
```
Access the app at http://localhost:8000 🎉

### Option B: Quick Start
>**WARNING:** The `main` branch no longer supports the barebones method of running the application. The steps below are for reference. For running the current application, use **Option A** with docker setup instead.

The steps to run the barebones Chainlit application.

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
    Set `PYTHONPATH` to `src` if not already set:
   ```bash
   export PYTHONPATH=./src
   ```
   and then verify using `echo`
6. List embeddings available for download:
    ```bash
    ./bin/embeddings_manager ls-remote
    ```
7. Install your chosen embeddings:
    ```bash
    ./bin/embeddings_manager install openai/text-embedding-3-large/reactome/ReleaseXX
    ```
8. Run the Chainlit application:
    ```
    chainlit run bin/chat-chainlit.py -w
    ```
9. Access the app at http://localhost:8000 🎉

## Embeddings & Documents Bundles

The ChatBot's knowledge of a given data source is generated using the latest data release, resulting in a bundle of embedded information and/or text documents. For simplicity, we refer to these bundles as **Embeddings** throughout this document.

In the case of Reactome, embeddings bundles are generated once per release from [reactome/graphdb](https://hub.docker.com/r/reactome/graphdb) releases from DockerHub and uploaded to AWS S3 for easy retrieval.

### Embeddings Manager Script

All aspects of generating, managing, uploading, and retrieving embeddings bundles are handled by the `./bin/embeddings_manager` script.
- Basic usage is covered in the **_Quick Start_** guide above.
- See the [Embeddings Manager documentation](docs/embeddings_manager.md) for more information.


## Troubleshooting

>**ISSUE: "The Docker build is taking an unusually long time"**
  - Use prebuilt docker image.

 
>**ISSUE: "Docker image: Build vs Prebuilt"**
  - Building the Docker image locally can take a significant amount of time - potentially hours, especially on Windows. If you are not making any changes to the codebase, i.e. modyfying any code, you can use the publicly available prebuilt image.

    
>**ISSUE: "Why do I get `{"detail":"Not Found"}` when I try to access the Chatbot at http://localhost:8000?"**
   - Use http://localhost:8000/chat
     
    
>**ISSUE: " `An error occurred (AccessDenied) when calling the ListObjects operation: Access Denied` when `./bin/embeddings_manager ls-remote` is called."**
  - If you don't have an AWS authentication use `install` command to fetch embeddings instead
    ```bash
    ./bin/embeddings_manager install openai/text-embedding-3-large/reactome/ReleaseXX
    ```
    Or install an embedding directly in the format `<modelorg>/<model>` specifying a model and a compatible released Reactome version.
    > Example: openai/text-embedding-3-large/reactome/Release89


## Developers

### Code Quality

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


### Contributing
Contributions to the Reactome ChatBot project are welcome! If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

Please make sure to follow our contributing guidelines and code of conduct.

## License

This project is licensed under the [MIT License](LICENSE).
