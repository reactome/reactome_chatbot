"""The two welcome screens, and which one the image ships.

chainlit renders `chainlit.md` from CHAINLIT_APP_ROOT at startup. /app is
read-only in the image, so a deployment selects its variant by mounting one over
the other -- the same mechanism config.yml already uses. That means the file
names and their contents are the interface, and this pins them.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
DEFAULT = REPO_ROOT / "chainlit.md"
PLANT = REPO_ROOT / "chainlit.plantreactome.md"


def test_both_welcome_pages_exist() -> None:
    assert DEFAULT.exists(), "the image ships this one"
    assert PLANT.exists(), "and this is mounted over it for Plant Reactome"


def test_the_shipped_default_is_the_reactome_one() -> None:
    """The image is built for Reactome; Plant Reactome overrides by mount.

    This was the other way round: main shipped the Plant Reactome text, so the
    Reactome deployment showed a welcome screen naming the wrong knowledgebase.
    """
    text = DEFAULT.read_text()
    assert "Plant Reactome" not in text
    assert "exploring Reactome" in text


def test_the_plant_variant_names_plant_reactome() -> None:
    assert "Plant Reactome" in PLANT.read_text()


def test_the_dockerfile_ships_the_default_and_not_the_variant() -> None:
    """Shipping both would make it ambiguous which one is in effect."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "COPY --chown=appuser --chmod=400 chainlit.md /app/" in dockerfile
    assert "chainlit.plantreactome.md" not in dockerfile


@pytest.mark.parametrize("page", [DEFAULT, PLANT])
def test_neither_page_is_empty(page: Path) -> None:
    """chainlit treats an empty chainlit.md as "no welcome screen"."""
    assert page.read_text().strip()
