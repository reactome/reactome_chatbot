import os
from pathlib import Path

# Define the base directory for embeddings
EM_ARCHIVE = Path(os.getenv("EM_ARCHIVE", Path.home() / ".cache" / "reactome" / "embeddings"))

class EmbeddingEnvironment:
    """Singleton-like class to manage the active embeddings directory."""
    
    _active_embedding: str | None = None
    _embedding_paths: dict[str, Path] = {}
    
    @classmethod
    def set_one(cls, relative_path: str) -> None:
        """Set the active embedding by relative path from EM_ARCHIVE."""
        cls._active_embedding = relative_path
    
    @classmethod
    def get_dir(cls, db_name: str = None) -> Path | None:
        """Get the directory for a specific database or the active one."""
        if db_name:
            # Construct path for specific database
            # This is a simplified version - in reality it would parse the embedding selection
            return EM_ARCHIVE / db_name
        elif cls._active_embedding:
            return EM_ARCHIVE / cls._active_embedding
        return None
    
    @classmethod
    def get_dict(cls) -> dict[str, str]:
        """Get dictionary of active embeddings (for compatibility)."""
        if cls._active_embedding:
            # Return a dict with the active embedding info
            parts = cls._active_embedding.split("/")
            if len(parts) >= 3:
                db_name = parts[2]  # modelorg/model/database/version
                return {db_name: str(EM_ARCHIVE / cls._active_embedding)}
        return {}

# Ensure the directory exists
EM_ARCHIVE.mkdir(parents=True, exist_ok=True)