"""Feature gating.

`postprocessing` is the external web search step, enabled for everyone in the
shipped config, so these two conditions decide whether a Tavily call happens.
"""

import pytest
from pydantic import ValidationError

from util.config_yml import Config
from util.config_yml.features import Feature, Features, UserGroup

LOGGED_IN = "someone@example.org"
GUEST = None


def test_user_group_all_matches_everyone() -> None:
    feature = Feature(enabled=True, user_group=UserGroup.all)
    assert feature.matches_user_group(LOGGED_IN) is True
    assert feature.matches_user_group(GUEST) is True


def test_user_group_logged_in_excludes_guests() -> None:
    feature = Feature(enabled=True, user_group=UserGroup.logged_in)
    assert feature.matches_user_group(LOGGED_IN) is True
    assert feature.matches_user_group(GUEST) is False


def test_omitted_user_group_matches_everyone() -> None:
    """user_group is optional in .config.schema.yaml; absent means unrestricted."""
    assert Feature(enabled=True).matches_user_group(GUEST) is True


def test_unknown_user_group_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Feature(enabled=True, user_group="admins")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("enabled", "user_group", "user_id", "expected"),
    [
        (True, "all", LOGGED_IN, True),
        (True, "all", GUEST, True),
        (True, "logged_in", LOGGED_IN, True),
        (True, "logged_in", GUEST, False),
        (False, "all", LOGGED_IN, False),
        (False, "logged_in", LOGGED_IN, False),
    ],
)
def test_get_feature_combines_enabled_and_group(
    enabled: bool, user_group: str, user_id: str | None, expected: bool
) -> None:
    config = Config(
        features=Features.model_validate(
            {"postprocessing": {"enabled": enabled, "user_group": user_group}}
        ),
        messages={},
        profiles=[],
        usage_limits={"message_rates": []},  # type: ignore[arg-type]
    )
    assert config.get_feature("postprocessing", user_id) is expected


def test_unknown_feature_id_defaults_to_enabled() -> None:
    """get_feature returns True for ids it does not know about.

    That is fail-open, so adding a call for a feature that is not in the model
    silently enables it for everyone rather than raising. Pinning the behaviour
    rather than endorsing it.
    """
    config = Config(
        features=Features.model_validate(
            {"postprocessing": {"enabled": False, "user_group": "all"}}
        ),
        messages={},
        profiles=[],
        usage_limits={"message_rates": []},  # type: ignore[arg-type]
    )
    assert config.get_feature("a_feature_that_does_not_exist") is True
