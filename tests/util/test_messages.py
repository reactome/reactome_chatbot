from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from util.config_yml.messages import Message, Trigger, TriggerEvent


def test_event_trigger_matches_only_its_own_event() -> None:
    trigger = Trigger(event=TriggerEvent.on_chat_start)
    assert trigger.match_trigger(TriggerEvent.on_chat_start) is True
    assert trigger.match_trigger(TriggerEvent.on_message) is False
    assert trigger.match_trigger(None) is False


def test_after_messages_fires_on_exact_equality_only() -> None:
    """Note: `!=`, not `<`. An `after_messages: 3` message fires on message 3 alone."""
    trigger = Trigger(after_messages=3)
    assert trigger.match_trigger(after_messages=3) is True
    assert trigger.match_trigger(after_messages=2) is False
    assert trigger.match_trigger(after_messages=4) is False


def test_start_and_end_bound_the_active_window() -> None:
    past = datetime.now() - timedelta(days=1)
    future = datetime.now() + timedelta(days=1)
    assert Trigger(start=past, end=future).match_trigger() is True
    assert Trigger(start=future).match_trigger() is False
    assert Trigger(end=past).match_trigger() is False


def test_timezone_aware_bounds_are_converted_not_stripped() -> None:
    """`Z` timestamps used to have tzinfo discarded rather than converted.

    config_default.yml writes bounds as `2025-01-01T00:00:00Z`. Stripping tzinfo
    shifted the window by the host's UTC offset -- invisible at day granularity,
    wrong at hour granularity. The comparison is now done in UTC on both sides.
    """
    now_utc = datetime.now(UTC)
    just_past = (now_utc - timedelta(minutes=5)).isoformat()
    just_future = (now_utc + timedelta(minutes=5)).isoformat()

    assert Trigger.model_validate({"start": just_past}).match_trigger() is True
    assert Trigger.model_validate({"start": just_future}).match_trigger() is False
    assert Trigger.model_validate({"end": just_future}).match_trigger() is True
    assert Trigger.model_validate({"end": just_past}).match_trigger() is False


def test_naive_bounds_are_treated_as_utc() -> None:
    """A bound written without an offset is assumed UTC rather than local."""
    past = (datetime.now(UTC) - timedelta(days=1)).replace(tzinfo=None)
    assert Trigger(end=past).match_trigger() is False
    assert Trigger(start=past).match_trigger() is True


def test_malformed_freq_max_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Trigger(event=TriggerEvent.on_message, freq_max="1min")


def test_freq_max_throttles_on_last_send_time() -> None:
    trigger = Trigger(event=TriggerEvent.on_message, freq_max="1m")
    just_now = datetime.now().isoformat()
    long_ago = (datetime.now() - timedelta(hours=1)).isoformat()
    assert (
        trigger.match_trigger(TriggerEvent.on_message, last_message=just_now) is False
    )
    assert trigger.match_trigger(TriggerEvent.on_message, last_message=long_ago) is True
    assert trigger.match_trigger(TriggerEvent.on_message, last_message=None) is True


def test_message_defaults_to_enabled_and_unrestricted() -> None:
    message = Message(message="hi", trigger=Trigger(event=TriggerEvent.on_chat_start))
    assert message.enabled is True
    assert message.recipients is None
