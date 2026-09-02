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
def test_malformed_intervals_raise(text: str) -> None:
    """Malformed intervals must be loud.

    This used to return timedelta(0), which silently disabled rate limiting: a
    zero-length window means every queued timestamp is already outside it, so the
    queue drained on every call and nobody was ever limited. Config fields now
    carry INTERVAL_PATTERN, so a bad value is rejected when config.yml loads and
    never reaches here -- see test_usage_limits.py.
    """
    with pytest.raises(ValueError, match="malformed interval"):
        parse_interval(text)
