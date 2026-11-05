"""The plantreactomedev override, checked without starting a container.

deploy/plantreactome/compose.plantreactome.yaml is the entire difference between this checkout and upstream's
compose.yaml, and until the deployment moved off docker-compose.yml nothing in the
test suite looked at it. Three of its failure modes are silent, which is why they
are pinned here rather than left to a smoke test:

  - `secrets: !override` replaces compose.yaml's secrets block *wholesale*. An
    eighth secret added upstream is not merged in and nothing warns about it;
    `docker compose up` just fails to resolve it, on someone else's host.
  - Every brand asset is a bind mount of a fixed filename. A name in the mount
    list with no file behind it is not an error either -- the engine creates an
    empty *directory* at that path, Chainlit's glob matches nothing, and it
    serves its own logo behind one INFO line.
  - Chainlit looks the message avatar up by a slug derived from `config.ui.name`
    (chainlit/server.py, `get_avatar`), so the avatar's filename is not a free
    choice, and a miss falls back to the favicon rather than to a failure.

None of this needs Docker, a network or a database: two YAML files, one TOML file
and a directory listing.
"""

import re
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from util.branding import PLANT_REACTOME, REACTOME

ROOT = Path(__file__).parent.parent.parent

PLACEHOLDER = "deploy/plantreactome/secrets/empty"
CALLER_PEM_PATH = "deploy/plantreactome/secrets/caller_token_public.pem"
#: The plantreactomedev override. Deliberately not named `deploy/plantreactome/compose.plantreactome.yaml`
#: and deliberately not in the project directory: Compose auto-discovers that
#: name next to the base file, so a merged copy would silently rewire a Reactome
#: `docker compose up`. Here it loads only when named explicitly.
OVERRIDE = Path("deploy/plantreactome/compose.plantreactome.yaml")

#: The prefix deploy/plantreactome/compose.plantreactome.yaml mounts brand assets from. BRAND_DIR is unset
#: for Reactome, which is why the default is spelled out in the key itself.
BRAND_MOUNT_PREFIX = "${BRAND_DIR:-./public}/"


#: A Compose merge tag sitting on a mapping or sequence key: `secrets: !override`.
#: Anchored to the end of a `key:` so a value that merely contains a bang -- a
#: generated password -- is left alone.
COMPOSE_TAG = re.compile(r"^(\s*[\w.-]+:)\s*!\S+", re.MULTILINE)


def load_compose(path: Path) -> dict[str, Any]:
    """Parse a Compose file, tolerating the merge tags.

    `yaml.safe_load` has no constructor for `!override`, which deploy/plantreactome/compose.plantreactome.yaml
    uses, and raises before any assertion runs. The tag says *how* a node merges
    into the other file -- not a question any test here asks. These ask which names
    are declared and to what path, so the tag is stripped and the value it carries
    is parsed as an ordinary mapping or sequence.

    Stripped as text rather than by teaching a `yaml.SafeLoader` subclass to
    construct unknown tags, because PyYAML is untyped here: subclassing it would
    trip mypy's `disallow_subclassing_any`, which is on repo-wide.
    """
    stripped = COMPOSE_TAG.sub(r"\1", path.read_text())
    # cast rather than a local: `yaml.safe_load` is typed to return Any, and
    # warn_return_any is on repo-wide, while assigning to an annotated local
    # first is what RET504 flags.
    return cast(dict[str, Any], yaml.safe_load(stripped))


def env_dev_value(name: str) -> str | None:
    """One line of .env.dev, or None. The file is gitignored, so it may not exist."""
    env_file = ROOT / ".env.dev"
    if not env_file.is_file():
        return None
    for line in env_file.read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip() or None
    return None


def avatar_slug(name: str) -> str:
    """`chainlit/server.py`'s get_avatar, verbatim.

    Not our rule, and that is the point of writing it down: the file Chainlit will
    look for is derived from `config.ui.name` by these four operations and no
    others, on a case-sensitive filesystem.
    """
    return name.strip().lower().replace(" ", "_").replace(".", "_")


# --- secrets ---------------------------------------------------------------


def test_the_override_declares_every_secret_any_service_references() -> None:
    """`!override` replaces the whole mapping, so upstream's additions vanish.

    This is the same silent drift tests/util/test_secrets.py pins on the other
    side of the same boundary, arriving from the other direction: there, a secret
    compose.yaml mounts that SECRET_NAMES does not list is mounted and never read;
    here, a secret compose.yaml adds that the override does not re-declare is
    referenced by a service and simply does not exist. `up` refuses to start.
    """
    base = load_compose(ROOT / "compose.yaml")
    override = load_compose(ROOT / OVERRIDE)

    referenced = {
        name
        for service in base["services"].values()
        for name in (service.get("secrets") or [])
    }
    declared_after_override = set(override["secrets"])

    missing = sorted(referenced - declared_after_override)
    assert not missing, (
        f"compose.yaml's services ask for {missing}, but deploy/plantreactome/compose.plantreactome.yaml's "
        "`secrets: !override` replaced the whole block without them. Add them."
    )


def test_no_secret_in_the_override_needs_a_swarm_manager() -> None:
    """The reason the override exists.

    `external: true` is resolved against the Swarm secret store, and a single
    engine is not a swarm manager, so one surviving entry takes the whole
    deployment down with "this node is not a swarm manager".
    """
    override = load_compose(ROOT / OVERRIDE)

    external = sorted(
        name
        for name, spec in (override["secrets"] or {}).items()
        if isinstance(spec, dict) and spec.get("external")
    )
    assert not external, f"still Swarm-only: {external}"


def test_the_placeholder_secret_is_present_and_empty() -> None:
    """Six of the seven secrets point at one committed empty file.

    It has to be *empty*, not merely blank: `mounted_secrets()` in
    src/util/secrets.py stats file size rather than reading contents, so a
    placeholder holding a single newline is logged as a supplied secret. And it
    has to exist, because it is the one part of the secrets directory that
    belongs in git -- `docker compose up` fails without it.
    """
    placeholder = ROOT / PLACEHOLDER

    assert (
        placeholder.is_file()
    ), f"{PLACEHOLDER} is committed and is not generated; it is missing."
    assert placeholder.stat().st_size == 0, "a newline would be logged as a secret"
    assert placeholder.read_text() == ""


@pytest.mark.skipif(
    not (ROOT / CALLER_PEM_PATH).is_file(),
    reason="generated by ./bin/make-caller-token-keypair.py, and gitignored",
)
def test_the_caller_token_key_is_readable_by_the_container_user() -> None:
    """File secrets are bind mounts, so the host's mode *is* the container's mode.

    Docker Compose ignores `mode:`/`uid:`/`gid:` for file-sourced secrets -- those
    are Swarm tmpfs features -- so nothing in the Compose file can rescue a key
    written 0600 by root. The image runs as appuser.
    """
    key = ROOT / CALLER_PEM_PATH

    assert key.stat().st_size > 0, "load_verifying_key() refuses to start on empty"
    assert key.stat().st_mode & 0o004, f"{key} must be world-readable"


# --- brand assets ----------------------------------------------------------


def brand_mounts() -> list[tuple[str, str]]:
    """(source relative to the brand tree, container target) for each brand mount.

    Parsed from the right of the prefix rather than by splitting the entry on `:`:
    the prefix is `${BRAND_DIR:-./public}/`, and a Compose default carries a colon
    of its own. `entry.split(":")[0]` returns `${BRAND_DIR`, nothing matches the
    prefix, and every test below quietly asserts against an empty list.
    """
    override = load_compose(ROOT / OVERRIDE)
    mounts: list[tuple[str, str]] = []
    for entry in override["services"]["chainlit"]["volumes"]:
        if not entry.startswith(BRAND_MOUNT_PREFIX):
            continue
        # `${BRAND_DIR:-./public}/logo_light.png:/app/public/logo_light.png:ro`
        # -- the filename is the first run after the prefix, and the target is
        # the run after that. Neither can contain a colon.
        after_prefix = entry[len(BRAND_MOUNT_PREFIX) :]
        source, _, rest = after_prefix.partition(":")
        target = rest.split(":")[0]
        mounts.append((source, target))
    return mounts


def test_the_brand_mounts_are_the_ones_chainlit_globs_for() -> None:
    """The mount list has to cover the names Chainlit actually asks for.

    /logo globs public/logo_{light,dark}.*, /favicon globs public/favicon.*, and
    avatars are a directory. A slot missing from the list is not an error at
    build time; it is the image's own Reactome logo, served from a Plant Reactome
    deployment.
    """
    sources = {source for source, _ in brand_mounts()}

    assert sources == {
        "logo_light.png",
        "logo_dark.png",
        "favicon.svg",
        "theme.json",
        "avatars",
    }, "the slots Chainlit resolves by fixed name, and nothing else"


def test_every_brand_mount_has_something_behind_it_in_every_tree() -> None:
    """One missing file becomes an empty directory, not a failure.

    Checked against every tree any deployment here mounts from: `public/`, which
    is what Reactome gets with BRAND_DIR unset and which must not be able to
    break, plus whatever BRAND_DIR names on this host.
    """
    trees = [Path("public")]
    brand_dir = env_dev_value("BRAND_DIR")
    if brand_dir:
        trees.append(Path(brand_dir.removeprefix("./")))

    for tree in trees:
        assert (ROOT / tree).is_dir(), f"{tree} is a mounted tree and does not exist"
        for source, target in brand_mounts():
            path = ROOT / tree / source
            if target == "/app/public/avatars":
                assert path.is_dir(), f"{path} is mounted as a directory"
                assert any(path.iterdir()), (
                    f"{path}: the mount replaces the image's whole avatars "
                    "directory, so an empty one deletes every avatar."
                )
            else:
                assert path.is_file(), (
                    f"{path} is mounted at {target} and is not a file. Docker "
                    "creates a directory there and Chainlit falls back to its own "
                    "asset, one INFO line from looking fine."
                )


def test_the_assistant_avatar_is_the_file_chainlit_will_look_for() -> None:
    """`bin/chat-chainlit.py` sets `config.ui.name`, and that sets the filename.

    The two halves have to agree or the assistant's avatar is the favicon --
    Chainlit's documented fallback, so it degrades instead of failing and nobody
    notices. Reactome's row is also the proof that setting `ui.name` is a no-op
    for the deployment that has always had it.
    """
    cases = [(REACTOME, Path("public"))]
    brand_dir = env_dev_value("BRAND_DIR")
    if brand_dir:
        cases.append((PLANT_REACTOME, Path(brand_dir.removeprefix("./"))))

    for branding, tree in cases:
        slug = avatar_slug(branding.chatbot_name)
        matches = list((ROOT / tree / "avatars").glob(f"{slug}.*"))
        assert matches, (
            f"{tree}/avatars/{slug}.* is what /avatars/default resolves to once "
            f"ui.name is {branding.chatbot_name!r}; found none, so readers of "
            "that deployment get the favicon as the assistant's avatar."
        )


def test_naming_the_reactome_chatbot_changes_nothing() -> None:
    """The Reactome no-op, pinned.

    bin/chat-chainlit.py overwrites `config.ui.name` with `BRANDING.chatbot_name`
    on every deployment, including Reactome's. That is only safe while the two
    values agree; if they ever differ, Reactome's tab title and avatar filename
    change from a code edit, with no config change and no deploy to point at.
    """
    toml = tomllib.loads((ROOT / ".chainlit" / "config.toml").read_text())

    assert toml["UI"]["name"] == REACTOME.chatbot_name
