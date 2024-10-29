from pathlib import Path

REPO_ROOT: Path = Path(__file__).parent.parent.parent
EM_ARCHIVE: Path = REPO_ROOT / "embeddings"
EM_CURRENT: Path = EM_ARCHIVE / "current"


class EmbeddingEnvironment:
    def __init__(self, env_path: str):
        self.embeddings: dict[str, Path] = dict()
        if env_path != "":
            for embedding_path in map(Path, env_path.split(":")):
                db: str = embedding_path.parent.name
                self.embeddings[db] = embedding_path

    @classmethod
    def _get(cls):  # -> Self
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
    def get_dir(cls, key: str) -> Path:
        return EM_ARCHIVE / cls._get().embeddings[key]

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
