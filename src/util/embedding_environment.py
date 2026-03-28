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
    def _get(cls):
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
        else:
            return None

    @classmethod
    def get_dir_or_raise(cls, key: str) -> Path:
        """
        Like get_dir(), but raises RuntimeError with actionable install
        instructions instead of returning None.

        Prevents downstream AttributeError: 'NoneType' object has no
        attribute 'glob' when embeddings are not installed.

        Raises:
            RuntimeError: if no embeddings are configured for `key`,
                          or if the configured directory does not exist on disk.
        """
        directory = cls.get_dir(key)
        if directory is None:
            raise RuntimeError(
                f"\n[ERROR] No embeddings configured for '{key}'.\n"
                f"Install them with:\n\n"
                f"  ./bin/embeddings_manager install "
                f"openai/text-embedding-3-large/{key}/ReleaseXX\n\n"
                f"List available versions with:\n"
                f"  ./bin/embeddings_manager ls-remote\n"
            )
        if not directory.exists():
            raise RuntimeError(
                f"\n[ERROR] Embeddings directory configured but missing on disk:\n"
                f"  {directory}\n\n"
                f"Re-install with:\n"
                f"  ./bin/embeddings_manager install "
                f"openai/text-embedding-3-large/{key}/ReleaseXX\n"
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