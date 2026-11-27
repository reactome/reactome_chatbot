"""The manifest of a three-collection Reactome bundle, and its reader.

Written by `data_generation.reactome.bundle_generator`; read by the retriever
and the Reactome answer chain. A bundle without one is an older bundle and is
read as before. A manifest that cannot be read raises, rather than being
treated as absent.
"""

import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST = "manifest.json"

#: Bumped whenever a reader would have to treat the manifest differently.
SCHEMA_VERSION = 1

#: Present while a bundle is being built. A bundle that still has it did not
#: finish, and has no manifest, so it would otherwise be read as an old one.
INCOMPLETE = ".incomplete"


@dataclass(frozen=True)
class CollectionLayout:
    """How one collection was built, which is how it must be read."""

    name: str
    embedded: tuple[str, ...]
    metadata: tuple[str, ...]
    shown_to_model: tuple[str, ...]
    skip_empty_fields: bool


@dataclass(frozen=True)
class BundleManifest:
    collections: tuple[CollectionLayout, ...]

    @property
    def shown_to_model(self) -> tuple[str, ...]:
        """Every field shown to the model by any collection, in first-seen order."""
        seen: dict[str, None] = {}
        for collection in self.collections:
            seen.update(dict.fromkeys(collection.shown_to_model))
        return tuple(seen)


def read_manifest(bundle_dir: Path) -> BundleManifest | None:
    """The bundle's manifest, None for an older bundle.

    Raises if the manifest is unreadable or the bundle did not finish building.
    """
    if (bundle_dir / INCOMPLETE).exists():
        raise ValueError(
            f"{bundle_dir} did not finish building. Rebuild it before using it."
        )
    path = bundle_dir / MANIFEST
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"schema version {version}, and this code reads version "
                f"{SCHEMA_VERSION}"
            )
        collections = tuple(
            CollectionLayout(
                name=name,
                embedded=tuple(layout["embedded"]),
                metadata=tuple(layout["metadata"]),
                shown_to_model=tuple(layout["shown_to_model"]),
                skip_empty_fields=bool(layout["skip_empty_fields"]),
            )
            for name, layout in data["collections"].items()
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError(
            f"{path} cannot be read as a bundle manifest: {error}"
        ) from error
    return BundleManifest(collections=collections)
