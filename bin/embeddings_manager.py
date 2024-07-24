from argparse import ArgumentParser
import os
from pathlib import Path
import re
from typing import NamedTuple
from zipfile import ZipFile, ZIP_DEFLATED

import localstack_client.session as boto3


REPO_ROOT = Path(__file__).parent.parent
S3_BUCKET = "embeddings"


class EmbeddingSelection(NamedTuple):
    model: str
    db: str
    version: str

    def __str__(self) -> str:
        return "/".join(self)

    def is_openai(self) -> bool:
        return self.model.startswith("openai/text-embedding-")

    def path(self, check_exists:bool=True) -> Path:
        path = REPO_ROOT / ".embeddings" / str(self)
        if check_exists and not path.is_dir():
            exit(f"Embedding does not exist at {path}")
        return path

    @classmethod
    def parse(cls, embedding_id:str): # -> Self
        embedding_id = embedding_id.replace("\\", "/")
        match = re.fullmatch(r"([^/]+/[^/]+)/([^/]+)/([^/]+)", embedding_id)
        if match is None:
            raise ValueError(f"malformed selection string '{embedding_id}'")
        return cls(*match.groups())


def pull(embedding: EmbeddingSelection):
    embedding_path:Path = embedding.path(check_exists=False)
    zip_tmpfile:Path = REPO_ROOT / ".embeddings/tmp.zip"
    try:
        s3 = boto3.client("s3")
        print("Downloading...")
        s3.download_file(S3_BUCKET, str(embedding), zip_tmpfile)
        print("Decompressing...")
        with ZipFile(zip_tmpfile, "r") as zipf:
            zipf.extractall(embedding_path)
    finally:
        zip_tmpfile.unlink(missing_ok=True)
    print(f"Saved to {embedding_path}")


def use(embedding: EmbeddingSelection):
    (REPO_ROOT / "embeddings").mkdir(exist_ok=True)
    embedding_path:Path = embedding.path()
    (REPO_ROOT / "embeddings" / embedding.db).unlink(missing_ok=True)
    (REPO_ROOT / "embeddings" / embedding.db).symlink_to(embedding_path)
    which()


def install(embedding: EmbeddingSelection):
    pull(embedding)
    use(embedding)


def make(
    embedding: EmbeddingSelection,
    openai_key: str = None,
    hf_key: str = None,
    **kwargs
):
    embedding_path:Path = embedding.path(check_exists=False)
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if hf_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_key
    if embedding.db == "reactome":
        from embeddings.reactome_generator import main
        main(str(embedding_path), hf_model=embedding.model, **kwargs)
    else:
        raise NotImplementedError(f"db: {embedding.db}")
    use(embedding)


def push(embedding: EmbeddingSelection):
    embedding_path:Path = embedding.path()
    zip_tmpfile:Path = REPO_ROOT / ".embeddings/tmp.zip"
    try:
        print("Compressing...")
        with ZipFile(zip_tmpfile, "w", ZIP_DEFLATED) as zipf:
            for root, _, filenames in os.walk(embedding_path):
                for filename in filenames:
                    file_path = Path(root) / filename
                    arc_name = file_path.relative_to(embedding_path)
                    zipf.write(file_path, arc_name)
        s3 = boto3.client("s3")
        print("Uploading...")
        s3.upload_file(zip_tmpfile, S3_BUCKET, str(embedding))
    finally:
        zip_tmpfile.unlink()
    print("Pushed to S3.")


def which():
    for subdir in (REPO_ROOT / "embeddings").iterdir():
        abs_path = subdir.resolve()
        try:
            display_path = abs_path.relative_to(REPO_ROOT / ".embeddings")
        except ValueError:
            display_path = abs_path
        print(f"{subdir.name}:\t{display_path}")


if __name__ == "__main__":
    parser = ArgumentParser()

    # Parent parser for selecting embeddings
    selection_parser = ArgumentParser(add_help=False)
    selection_parser.add_argument(
        "embedding",
        type=EmbeddingSelection.parse,
        help="Embedding selection: <modelorg>/<model>/<database>/<version>"
    )

    # Subcommands
    subparsers = parser.add_subparsers(required=True)
    pull_parser = subparsers.add_parser(
        "pull",
        parents=[selection_parser],
        help="Download embeddings",
    )
    pull_parser.set_defaults(func=pull)
    use_parser = subparsers.add_parser(
        "use",
        parents=[selection_parser],
        help="Set the active embeddings",
    )
    use_parser.set_defaults(func=use)
    install_parser = subparsers.add_parser(
        "install",
        parents=[selection_parser],
        help="Download and set the active embeddings (pull+use)",
    )
    install_parser.set_defaults(func=install)
    make_parser = subparsers.add_parser(
        "make",
        parents=[selection_parser],
        help="Generate embeddings",
    )
    make_parser.set_defaults(func=make)
    push_parser = subparsers.add_parser(
        "push",
        parents=[selection_parser],
        help="Upload embeddings",
    )
    push_parser.set_defaults(func=push)
    rm_parser = subparsers.add_parser(
        "rm",
        parents=[selection_parser],
        help="Remove specified embedding (locally)",
    )
    ls_parser = subparsers.add_parser(
        "ls",
        help="List locally installed embeddings",
    )
    ls_remote_parser = subparsers.add_parser(
        "ls-remote",
        help="List available embeddings on S3",
    )
    which_parser = subparsers.add_parser(
        "which",
        help="Reveal the current embeddings in use",
    )
    which_parser.set_defaults(func=which)

    # Command-specific arguments
    make_parser.add_argument(
        "--openai-key",
        help="API key for OpenAI"
    )
    make_parser.add_argument(
        "--hf-key",
        help="API key for HuggingFaceHub",
    )
    make_parser.add_argument(
        "--device",
        help="PyTorch device to use when running HuggingFace model locally [cpu/cuda]",
    )
    make_parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7687",
        help="URI for Neo4j database connection",
    )
    make_parser.add_argument(
        "--neo4j-username",
        help="Username for Neo4j database connection",
    )
    make_parser.add_argument(
        "--neo4j-password",
        help="Password for Neo4j database connection",
    )
    make_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of CSV files"
    )

    args = parser.parse_args()
    func = args.func
    delattr(args, "func")
    func(**vars(args))
