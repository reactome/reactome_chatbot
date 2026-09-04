"""Config loading, with an emphasis on what happens when config.yml is wrong.

`Config.from_yaml` returning None switches off every config-driven feature,
including the message quota, so the fallback path matters as much as the happy
one: a typo in config.yml must not quietly remove rate limiting.
"""

from pathlib import Path

import pytest

from util.config_yml import CONFIG_DEFAULT_YML, CONFIG_YML, Config

VALID = """
profiles: ["React-to-Me"]
features:
  postprocessing:
    enabled: true
    user_group: all
usage_limits:
  message_rates:
    - users: ["all"]
      max_messages: 5
      interval: 1h
messages:
  hello:
    message: hi
    trigger:
      event: on_chat_start
"""


def test_the_shipped_default_config_is_valid() -> None:
    """If this fails, every fallback below lands on None and features silently die."""
    config = Config.from_yaml(CONFIG_DEFAULT_YML)
    assert config is not None
    assert config.usage_limits.message_rates, "defaults must carry a message quota"


def test_valid_config_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(VALID)
    config = Config.from_yaml(path)
    assert config is not None
    assert config.usage_limits.message_rates[0].max_messages == 5


def test_missing_config_falls_back_to_defaults(tmp_path: Path) -> None:
    config = Config.from_yaml(tmp_path / "nope.yml")
    assert config is not None
    assert config.usage_limits.message_rates


def test_invalid_config_refuses_to_start(tmp_path: Path) -> None:
    """A present-but-invalid config.yml must be fatal, not silently substituted.

    Both quiet options are wrong. Returning None disables every config-driven
    feature including the quota, so one typo removed rate limiting. Falling back
    to the defaults silently applies settings nobody chose -- see
    test_fallback_would_have_re_enabled_a_disabled_feature for why that matters.
    """
    path = tmp_path / "config.yml"
    path.write_text(VALID.replace("interval: 1h", "interval: 1hr"))
    with pytest.raises(SystemExit, match="Invalid config"):
        Config.from_yaml(path)


def test_fallback_would_have_re_enabled_a_disabled_feature(tmp_path: Path) -> None:
    """Why the fallback was the wrong fix, pinned so it is not reintroduced.

    An operator who turns postprocessing off and makes an unrelated typo would,
    under a silent fallback, get the default config back -- which has
    postprocessing enabled and a quota of 100. Enabling external web search
    because of a typo elsewhere in the file is a cost and privacy change nobody
    asked for.
    """
    disabled = VALID.replace("enabled: true", "enabled: false")
    assert "enabled: false" in disabled

    good = tmp_path / "good.yml"
    good.write_text(disabled)
    config = Config.from_yaml(good)
    assert config is not None
    assert config.features.postprocessing.enabled is False

    # the same file with a typo must now raise rather than quietly flip it back on
    bad = tmp_path / "bad.yml"
    bad.write_text(disabled.replace("interval: 1h", "interval: 1hr"))
    with pytest.raises(SystemExit):
        Config.from_yaml(bad)

    defaults = Config.from_yaml(CONFIG_DEFAULT_YML)
    assert defaults is not None
    assert (
        defaults.features.postprocessing.enabled is True
    ), "the default this would have silently substituted"


def test_config_yml_as_a_directory_falls_back(tmp_path: Path) -> None:
    """docker-compose bind-mounts ./config.yml; if it is absent on the host,
    Docker creates a *directory* there. That means "no config supplied", not
    "broken config", so it falls back rather than refusing to start."""
    path = tmp_path / "config.yml"
    path.mkdir()
    assert Config.from_yaml(path) is not None


@pytest.mark.parametrize("content", ["", "\n", "just a string", "[1, 2, 3]"])
def test_non_mapping_config_is_fatal(tmp_path: Path, content: str) -> None:
    """A file that exists but is not a config mapping is a mistake, not a default."""
    path = tmp_path / "config.yml"
    path.write_text(content)
    with pytest.raises(SystemExit):
        Config.from_yaml(path)


def test_unknown_profile_is_fatal(tmp_path: Path) -> None:
    """Starting with a profile the agent cannot build should not be silent."""
    path = tmp_path / "config.yml"
    path.write_text(VALID.replace('["React-to-Me"]', '["Nonexistent Profile"]'))
    with pytest.raises(SystemExit):
        Config.from_yaml(path)


def test_config_paths_do_not_depend_on_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defaults must resolve from anywhere, not just the repo root.

    These used to be Path("config.yml") / Path("config_default.yml"), i.e.
    relative to the process working directory, so running from a subdirectory
    silently lost the config.
    """
    assert CONFIG_YML.is_absolute()
    assert CONFIG_DEFAULT_YML.is_absolute()
    monkeypatch.chdir(tmp_path)
    assert Config.from_yaml(CONFIG_DEFAULT_YML) is not None
