"""The captcha pass: expires on the server, renews in use, names one solve."""

import hashlib
import hmac

from util import captcha_cookie as cookie

SECRET = "test-secret"  # noqa: S105
NOW = 1_800_000_000.0


def test_a_fresh_pass_checks_out_and_names_its_solve() -> None:
    verdict = cookie.check(cookie.mint(SECRET, NOW), SECRET, NOW)
    assert verdict.ok
    assert len(verdict.nonce) >= 16
    assert not verdict.renew


def test_two_solves_have_different_nonces() -> None:
    a = cookie.check(cookie.mint(SECRET, NOW), SECRET, NOW)
    b = cookie.check(cookie.mint(SECRET, NOW), SECRET, NOW)
    assert a.nonce != b.nonce


def test_it_expires_on_the_server_whatever_the_browser_was_told() -> None:
    minted = cookie.mint(SECRET, NOW)
    assert cookie.check(minted, SECRET, NOW + cookie.MAX_AGE_SECONDS - 1).ok
    assert not cookie.check(minted, SECRET, NOW + cookie.MAX_AGE_SECONDS + 1).ok


def test_it_asks_to_be_renewed_while_in_use() -> None:
    minted = cookie.mint(SECRET, NOW)
    assert cookie.check(minted, SECRET, NOW + cookie.RENEW_AFTER_SECONDS + 1).renew


def test_the_old_timeless_format_is_refused_even_when_signed() -> None:
    # value|HMAC(value): a pass that never expired (review, area 1b).
    value = "turnstile-token"
    signed = f"{value}|{hmac.new(SECRET.encode(), value.encode(), hashlib.sha256).hexdigest()}"
    assert not cookie.check(signed, SECRET, NOW).ok


def test_tampering_and_other_secrets_fail() -> None:
    minted = cookie.mint(SECRET, NOW)
    value, _, signature = minted.partition("|")
    later = value.replace(value.split(".")[1], str(int(NOW) + 10_000))
    assert not cookie.check(f"{later}|{signature}", SECRET, NOW + 10_000).ok
    assert not cookie.check(minted, "another-secret", NOW).ok
    assert not cookie.check(minted, "", NOW).ok


def test_a_pass_from_far_in_the_future_fails() -> None:
    minted = cookie.mint(SECRET, NOW + 3600)
    assert not cookie.check(minted, SECRET, NOW).ok
    assert cookie.check(cookie.mint(SECRET, NOW + 30), SECRET, NOW).ok


def test_garbage_never_raises() -> None:
    for junk in (None, "", "|", "a|b|c", "v2.x.y|sig", "\x00"):
        assert not cookie.check(junk, SECRET, NOW).ok


def test_reading_it_from_a_cookie_header() -> None:
    header = f"other=1; {cookie.COOKIE_NAME}=v2.1.abc|def; x=y"
    assert cookie.from_cookie_header(header) == "v2.1.abc|def"
    assert cookie.from_cookie_header(None) is None
    assert cookie.from_cookie_header("other=1") is None
