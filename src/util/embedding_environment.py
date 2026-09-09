from pathlib import Path

REPO_ROOT: Path = Path(__file__).parent.parent.parent
EM_ARCHIVE: Path = REPO_ROOT / "embeddings"
EM_CURRENT: Path = EM_ARCHIVE / "current"


class EmbeddingEnvironment:
    def __init__(self, env_path: str) -> None:
        self.embeddings: dict[str, Path] = {}
        if env_path != "":
            for embedding_path in map(Path, env_path.split(":")):
                db: str = embedding_path.parent.name
                self.embeddings[db] = embedding_path

    @classmethod
    def _get(cls) -> "EmbeddingEnvironment":
        if EM_CURRENT.exists():
            with EM_CURRENT.open("r") as current_fp:
                env_path = current_fp.read()
        else:
            env_path = ""
        return cls(env_path)

    @classmethod
    def get_dict(cls) -> dict[str, Path]:
        return cls._get().embeddings

    @classmethod
    def get_dir(cls, key: str) -> Path | None:
        if key in cls._get().embeddings:
            return EM_ARCHIVE / cls._get().embeddings[key]
        return None

    @classmethod
    def require_dir(cls, key: str) -> Path:
        """`get_dir`, but for bundles the caller cannot run without.

        `get_dir` returns None for an unknown key, and returns a path for a
        known one without checking that the path exists. Both failures used to
        travel: None was passed into a parameter typed `Path` (four mypy
        baseline entries suppressed the error), and a stale `embeddings/current`
        pointing at a deleted bundle handed Chroma a missing directory, which
        Chroma creates -- so the chatbot answered every question from an empty
        collection instead of refusing to start.

        Raising FileNotFoundError rather than SystemExit: it is the accurate
        exception, nothing catches it for a required bundle so the process still
        stops, and the optional userguide path in react_to_me.py already catches
        exactly this to degrade deliberately.
        """
        directory = cls.get_dir(key)
        if directory is None:
            available = ", ".join(sorted(cls.get_dict())) or "none"
            raise FileNotFoundError(
                f"No embeddings bundle installed for {key!r} (installed: {available}). "
                f"Run ./bin/embeddings_manager install <embedding-id>."
            )
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{key!r} points at {directory}, which does not exist. "
                f"{EM_CURRENT} is stale; re-run ./bin/embeddings_manager install."
            )
        return directory

    @classmethod
    def get_model(cls, key: str) -> str:
        return str(cls._get().embeddings[key].parent.parent)

    @classmethod
    def set_one(cls, embedding_path: Path) -> None:
        db: str = embedding_path.parent.name
        embeddings_dict: dict[str, Path] = cls.get_dict()
        embeddings_dict[db] = embedding_path
        env_path: str = ":".join(map(str, embeddings_dict.values()))
        with EM_CURRENT.open("w") as current_fp:
            current_fp.write(env_path)
