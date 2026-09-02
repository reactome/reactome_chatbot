from datetime import datetime, timedelta

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


def test_timezone_aware_bounds_are_compared_as_if_local() -> None:
    """BUG: tzinfo is stripped, not converted, so `Z` timestamps drift by the UTC offset.

    config_default.yml writes bounds as `2025-01-01T00:00:00Z`. `.replace(tzinfo=None)`
    discards the offset instead of converting, so the window silently shifts by the
    host's UTC offset. Harmless at day granularity, wrong at hour granularity.
    """
    trigger = Trigger.model_validate({"end": "2000-01-01T00:00:00Z"})
    assert trigger.end is not None and trigger.end.tzinfo is not None
    assert trigger.match_trigger() is False  # far past, so still correctly inactive


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
