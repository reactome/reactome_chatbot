# Reactome Chatbot Embeddings Manager Script

The script located at `bin/embeddings_manager.py` handles the generation, version-switching, and S3 upload/download of embeddings for the chatbot. Embeddings are installed locally to the `embeddings/` directory.

```
$ python .\bin\embeddings_manager.py -h
usage: embeddings_manager.py [-h] {pull,use,install,make,push,rm,ls,ls-remote,which} ...

positional arguments:
  {pull,use,install,make,push,rm,ls,ls-remote,which}
    pull                Download embeddings
    use                 Set the active embeddings
    install             Download and set the active embeddings (pull+use)
    make                Generate embeddings
    push                Upload embeddings
    rm                  Remove specified embedding (locally)
    ls                  List locally installed embeddings
    ls-remote           List available embeddings on S3
    which               Reveal the current embeddings in use

optional arguments:
  -h, --help            show this help message and exit
```

## Embedding Identifiers:

The script specifies embeddings using strings with the following format:
```
<modelorg>/<model>/<database>/<version>
```

For example, the embeddings for Reactome Release89 using the default [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings/embedding-models) model (`text-embedding-ada-002`) are specified as:
```
openai/text-embedding-ada-002/reactome/Release89
```

For HuggingFace models, `<modelorg>/<model>` simply matches the HuggingFace model identifier.

## Downloading Embeddings from S3

### List embeddings available to download: `ls-remote`
```sh
python bin/embeddings_manager.py ls-remote
```

### Download embedding: `pull`
```sh
python bin/embeddings_manager.py pull openai/text-embedding-ada-002/reactome/Release89
```

## Managing Local Embeddings:

### List installed embeddings: `ls`
```sh
python bin/embeddings_manager.py ls
```

### Select an embedding for use: `use`
```sh
python bin/embeddings_manager.py use openai/text-embedding-ada-002/reactome/Release89
```

### Check the current embeddings in use: `which`
```
$ python bin/embeddings_manager.py which
reactome:       nomic-ai\nomic-embed-text-v1.5\reactome\Release89
alliance:       ...
...
```

## Generate Embeddings: `make`

### Reactome:

#### Requirements:

- Reactome Dockerized Graph database from DockerHub: [reactome/graphdb](https://hub.docker.com/r/reactome/graphdb)
    + Be sure to note the Release# in use.

#### OpenAi generation (remote):
```sh
python bin/embeddings_manager.py make openai/text-embedding-ada-002/reactome/<Release#> --openai-key <your-key>
```

#### HuggingFace generation (local):
```sh
python bin/embeddings_manager.py make <hf-model>/reactome/<Release#> --device <cpu/cuda>
```

#### HuggingFaceHub generation (remote):
Either specify `--hf-key` or environment variable `HUGGINGFACEHUB_API_TOKEN`.
```sh
python bin/embeddings_manager.py make <hf-model>/reactome/<Release#> --hf-key <your-key>
```

## Uploading to S3: `push`

⚠️ Requires S3 write access.

```sh
python bin/embeddings_manager.py push openai/text-embedding-ada-002/reactome/Release89
```