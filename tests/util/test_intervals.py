from datetime import timedelta

import pytest

from util.config_yml.intervals import parse_interval


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30s", timedelta(seconds=30)),
        ("1m", timedelta(minutes=1)),
        ("3h", timedelta(hours=3)),  # the production usage_limits interval
        ("7d", timedelta(days=7)),
        ("2w", timedelta(weeks=2)),
        ("0s", timedelta(0)),
        ("100h", timedelta(hours=100)),
    ],
)
def test_parses_each_supported_unit(text: str, expected: timedelta) -> None:
    assert parse_interval(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "h", "3", "3x", "3 h", "-1h", "1.5h", "3H", "3hh", "1h30m"],
)
def test_malformed_intervals_degrade_to_zero(text: str) -> None:
    """BUG: malformed intervals fail silently as a zero-length window.

    `.config.schema.yaml` constrains `interval` to `^[0-9]+[smhdw]$`, but nothing
    validates config.yml against that schema at runtime, so a typo here reaches
    `MessageRate.check_rate` as timedelta(0) -- which drains the whole queue every
    call and silently disables rate limiting. See
    `test_zero_interval_silently_disables_rate_limiting`.
    """
    assert parse_interval(text) == timedelta(0)
