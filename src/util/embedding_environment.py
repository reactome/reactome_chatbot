from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
EM_ARCHIVE = REPO_ROOT / "embeddings"
EM_CURRENT = EM_ARCHIVE / "current"


class EmbeddingEnvironment:
    def __init__(self, env_path:str):
        self.embeddings:dict[str, Path] = dict()
        if env_path != "":
            for embedding_path in map(Path, env_path.split(":")):
                db = embedding_path.parent.name
                self.embeddings[db] = embedding_path

    @classmethod
    def _get(cls): # -> Self
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
    def set_one(cls, embedding_path:Path) -> None:
        db = embedding_path.parent.name
        embeddings_dict = cls.get_dict()
        embeddings_dict[db] = embedding_path
        env_path = ":".join(map(str, embeddings_dict.values()))
        with EM_CURRENT.open("w") as current_fp:
            current_fp.write(env_path)
