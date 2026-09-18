"""The answer endpoint's backstop limit (FR-008)."""

import time

import pytest

from util.rate_limit import SlidingWindowLimiter, identity_of, limiter_from_env


def test_requests_up_to_the_limit_are_allowed_and_the_next_is_not() -> None:
    limiter = SlidingWindowLimiter(limit=3, window=60.0)
    assert [limiter.allow("someone") for _ in range(3)] == [True, True, True]
    assert limiter.allow("someone") is False


def test_one_caller_over_the_limit_does_not_block_another() -> None:
    """The limit is per caller. A shared counter would let one visitor mute a page."""
    limiter = SlidingWindowLimiter(limit=1, window=60.0)
    assert limiter.allow("first") is True
    assert limiter.allow("first") is False
    assert limiter.allow("second") is True


def test_the_window_slides(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1000.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    limiter = SlidingWindowLimiter(limit=2, window=10.0)
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    assert limiter.allow("k") is False

    now = 1011.0  # Past the window, so the earlier hits no longer count.
    assert limiter.allow("k") is True


def test_stale_keys_are_evicted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without eviction the dict grows by one entry per visitor, forever."""
    now = 0.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    limiter = SlidingWindowLimiter(limit=5, window=10.0)
    for index in range(500):
        limiter.allow(f"visitor-{index}")
    assert len(limiter._hits) == 500

    now = 100.0  # Everything is stale; the next call sweeps.
    limiter.allow("someone-new")
    assert len(limiter._hits) == 1


def test_an_active_key_survives_a_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eviction must not reset the count of someone still calling."""
    now = 0.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    limiter = SlidingWindowLimiter(limit=2, window=10.0)
    # Both hits sit inside the window as measured at t=11, which covers (1, 11].
    # An earlier version of this test put one at t=0, which had genuinely expired
    # by then -- so it asserted a refusal the sliding window was right to allow.
    now = 5.0
    limiter.allow("busy")
    now = 9.0
    limiter.allow("busy")
    now = 11.0  # First sweep falls here; "busy" must come through it intact.
    assert limiter.allow("busy") is False, "the sweep reset an active caller's count"


class TestIdentity:
    def test_sub_is_preferred_when_the_website_sends_one(self) -> None:
        assert identity_of({"sub": "person-1"}, "tok") == "sub:person-1"

    def test_jti_is_used_when_there_is_no_sub(self) -> None:
        assert identity_of({"jti": "issue-1"}, "tok") == "jti:issue-1"

    def test_without_either_it_falls_back_to_the_token(self) -> None:
        """D1 is unsettled, so a token may carry nothing but `exp`."""
        first = identity_of({"exp": 1}, "token-a")
        second = identity_of({"exp": 1}, "token-b")
        assert first != second
        assert identity_of({"exp": 1}, "token-a") == first

    def test_the_raw_token_never_appears_in_the_key(self) -> None:
        """It is a credential and this key lives as long as the process."""
        token = "eyJhbGciOiJFZERTQSJ9.not-a-real-credential"  # noqa: S105
        assert token not in identity_of({}, token)

    def test_a_non_string_claim_is_ignored(self) -> None:
        """A token is attacker-influenced; `sub: {...}` must not become a key."""
        assert identity_of({"sub": {"nested": "x"}}, "tok").startswith("token:")


class TestConfiguration:
    def test_defaults_apply_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANSWER_RATE_LIMIT", raising=False)
        monkeypatch.delenv("ANSWER_RATE_WINDOW_SECONDS", raising=False)
        limiter = limiter_from_env()
        assert (limiter.limit, limiter.window) == (30, 600.0)

    @pytest.mark.parametrize("value", ["", "   ", "not-a-number", "0", "-5"])
    def test_unusable_configuration_falls_back_rather_than_disabling_the_limit(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        """A limit of 0 would refuse everyone; a crash would take the service down."""
        monkeypatch.setenv("ANSWER_RATE_LIMIT", value)
        assert limiter_from_env().limit == 30
