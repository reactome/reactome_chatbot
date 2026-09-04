"""Characterization of the production rate limiter.

config_default.yml ships `max_messages: 100` / `interval: 3h` for `users: ["all"]`,
so this is the code path that decides whether a real user gets an answer.
"""

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from util.config_yml.usage_limits import MessageRate, UsageLimits


def _ago(**kwargs: float) -> str:
    return (datetime.now() - timedelta(**kwargs)).isoformat()


def _rate(max_messages: int = 3, interval: str = "1h") -> MessageRate:
    return MessageRate(users=["all"], max_messages=max_messages, interval=interval)


def test_under_the_limit_is_allowed_and_records_the_message() -> None:
    queue = [_ago(minutes=5)]
    assert _rate().check_rate(queue) is None
    assert len(queue) == 2, "the allowed message must be recorded in the queue"


def test_at_the_limit_is_blocked_and_does_not_record() -> None:
    queue = [_ago(minutes=m) for m in (5, 10, 15)]
    rate = _rate(max_messages=3)
    assert rate.check_rate(queue) is rate, "returns itself to signal rate-limited"
    assert len(queue) == 3, "a blocked message must not consume quota"


def test_check_rate_mutates_the_callers_queue() -> None:
    """This in-place mutation is the contract `message_rate_limited` relies on.

    `util.chainlit_helpers.message_rate_limited` reads the queue out of user
    metadata, passes it in, and writes the *same list object* back. If check_rate
    ever stopped mutating, quota tracking would silently stop working.
    """
    queue: list[str] = []
    original = queue
    _rate().check_rate(queue)
    assert queue is original
    assert len(queue) == 1


def test_entries_older_than_the_interval_are_evicted() -> None:
    queue = [_ago(hours=5), _ago(hours=4), _ago(minutes=1)]
    assert _rate(max_messages=3, interval="1h").check_rate(queue) is None
    # the two stale entries are dropped, the recent one survives, the new one is added
    assert len(queue) == 2


def test_eviction_stops_at_the_first_in_window_entry() -> None:
    """The purge loop breaks on the first fresh entry rather than scanning the rest.

    The queue is only ever appended to in chronological order, so this is sound --
    but it means an out-of-order queue would retain stale entries.
    """
    queue = [_ago(minutes=1), _ago(hours=5)]
    _rate(max_messages=9, interval="1h").check_rate(queue)
    assert len(queue) == 3, "the stale entry behind a fresh one is not evicted"


@pytest.mark.parametrize("bad", ["3hr", "3", "h", "", "1h30m", "-1h"])
def test_malformed_interval_is_rejected_at_construction(bad: str) -> None:
    """A typo in config.yml must fail loudly instead of disabling the limiter.

    parse_interval() used to return timedelta(0) for anything unparseable, which
    made the window zero-length: the queue drained on every call and no user was
    ever limited. The field now carries the same pattern .config.schema.yaml
    documents, so the config is rejected at load time instead.
    """
    with pytest.raises(ValidationError):
        MessageRate(users=["all"], max_messages=1, interval=bad)


def test_max_messages_must_be_positive() -> None:
    """max_messages: 0 would block everyone; treat it as a config error."""
    with pytest.raises(ValidationError):
        MessageRate(users=["all"], max_messages=0, interval="1h")


def test_production_interval_is_accepted() -> None:
    assert _rate(max_messages=100, interval="3h").interval == "3h"


def test_first_matching_rule_wins() -> None:
    limits = UsageLimits(
        message_rates=[
            MessageRate(users=["all"], max_messages=1, interval="1h"),
            MessageRate(users=["logged_in"], max_messages=100, interval="1h"),
        ]
    )
    assert limits.message_rates[0].max_messages == 1
