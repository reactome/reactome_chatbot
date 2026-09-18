"""Every failure path refuses. There is no branch that lets an unverified token past.

The website mints these after its own captcha and proxies every request presenting
one; this service verifies a signature. Asymmetric on purpose -- they hold the
signing key, this holds only the public one -- because this is the side reachable
from a search page if the proxy is ever bypassed, and compromising it must not
produce valid tokens.
"""

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from util.caller_token import (
    ALGORITHMS,
    AUDIENCE_ENV,
    DEFAULT_AUDIENCE,
    KEY_PATH_ENV,
    TokenRejectedError,
    expected_audience,
    load_verifying_key,
    verify,
)


@pytest.fixture(scope="module")
def keys() -> tuple[str, str]:
    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = (
        private.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private_pem, public_pem


def _mint(private_pem: str, **claims: object) -> str:
    """The shape the website mints: aud included unless a test overrides it."""
    payload: dict[str, object] = {
        "exp": int(time.time()) + 300,
        "aud": DEFAULT_AUDIENCE,
    }
    payload.update(claims)
    payload = {k: v for k, v in payload.items() if v is not None}
    return jwt.encode(payload, private_pem, algorithm="EdDSA")


def test_a_validly_signed_token_is_accepted(keys: tuple[str, str]) -> None:
    private_pem, public_pem = keys
    claims = verify(_mint(private_pem, sub="visitor-1"), public_pem)
    assert claims["sub"] == "visitor-1"


def test_an_expired_token_is_refused(keys: tuple[str, str]) -> None:
    private_pem, public_pem = keys
    token = _mint(private_pem, exp=int(time.time()) - 1)
    with pytest.raises(TokenRejectedError, match="expired"):
        verify(token, public_pem)


def test_a_token_with_no_expiry_is_refused(keys: tuple[str, str]) -> None:
    """A token without one is a permanent credential."""
    private_pem, public_pem = keys
    token = _mint(private_pem, sub="forever", exp=None)
    with pytest.raises(TokenRejectedError, match="missing a required claim: exp"):
        verify(token, public_pem)


def test_a_token_for_another_audience_is_refused(keys: tuple[str, str]) -> None:
    """Enforcing `aud` is what stops a token minted for this service being
    replayed at a different consumer, and one minted elsewhere being presented
    here. The website asked for it explicitly."""
    private_pem, public_pem = keys
    token = _mint(private_pem, aud="some-other-service")
    with pytest.raises(TokenRejectedError, match="different audience"):
        verify(token, public_pem)


def test_a_token_with_no_audience_is_refused(keys: tuple[str, str]) -> None:
    """Refused by name, rather than being waved through as "nothing to check"."""
    private_pem, public_pem = keys
    token = _mint(private_pem, aud=None)
    with pytest.raises(TokenRejectedError, match="missing a required claim: aud"):
        verify(token, public_pem)


def test_the_expected_audience_is_configurable_but_never_empty(
    monkeypatch: pytest.MonkeyPatch, keys: tuple[str, str]
) -> None:
    """An empty setting must not mean "accept anything".

    PyJWT rejects a token carrying `aud` when none is expected, so a blank value
    would refuse every real token rather than loosening the check -- a
    misconfiguration that would look like a signing problem. It falls back to the
    agreed audience instead.
    """
    private_pem, public_pem = keys
    monkeypatch.setenv(AUDIENCE_ENV, "   ")
    assert expected_audience() == DEFAULT_AUDIENCE
    verify(_mint(private_pem), public_pem)  # still accepted

    monkeypatch.setenv(AUDIENCE_ENV, "someone-else")
    assert expected_audience() == "someone-else"
    with pytest.raises(TokenRejectedError, match="different audience"):
        verify(_mint(private_pem), public_pem)


def test_a_token_signed_by_someone_else_is_refused(keys: tuple[str, str]) -> None:
    _, public_pem = keys
    other = (
        Ed25519PrivateKey.generate()
        .private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        .decode()
    )
    with pytest.raises(TokenRejectedError):
        verify(_mint(other, sub="impostor"), public_pem)


def test_a_tampered_payload_is_refused(keys: tuple[str, str]) -> None:
    private_pem, public_pem = keys
    header, payload, signature = _mint(private_pem, sub="a").split(".")
    tampered = f"{header}.{payload[:-2]}XY.{signature}"
    with pytest.raises(TokenRejectedError):
        verify(tampered, public_pem)


@pytest.mark.parametrize("token", ["", "not-a-jwt", "a.b.c"])
def test_rubbish_is_refused(token: str, keys: tuple[str, str]) -> None:
    _, public_pem = keys
    with pytest.raises(TokenRejectedError):
        verify(token, public_pem)


def test_a_symmetric_token_is_refused(keys: tuple[str, str]) -> None:
    """HS256 signed with the public key as a secret.

    The classic JWT confusion attack: if HS* were accepted, anyone holding the
    public verifying key -- which is not a secret -- could mint valid tokens.
    """
    _, public_pem = keys

    # Hand-rolled rather than via jwt.encode, which refuses to HS256-sign PEM key
    # material. An attacker would not be using PyJWT, and the guard that matters
    # is on the verifying side.
    def b64(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    header = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = b64(json.dumps({"exp": int(time.time()) + 300, "sub": "forged"}).encode())
    signature = b64(
        hmac.new(
            public_pem.encode(), f"{header}.{payload}".encode(), hashlib.sha256
        ).digest()
    )
    with pytest.raises(TokenRejectedError):
        verify(f"{header}.{payload}.{signature}", public_pem)


def test_no_symmetric_algorithm_is_accepted_at_all() -> None:
    """The protection, stated directly rather than only through one forged token."""
    assert not [a for a in ALGORITHMS if a.startswith("HS")]


def test_a_missing_key_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Principle IV: accepting everything because the key is absent would test clean."""
    monkeypatch.delenv(KEY_PATH_ENV, raising=False)
    with pytest.raises(RuntimeError, match="cannot run without a verifying key"):
        load_verifying_key()


def test_an_empty_key_file_refuses_to_start(tmp_path: Path) -> None:
    empty = tmp_path / "key.pem"
    empty.write_text("   \n")
    with pytest.raises(RuntimeError, match="empty"):
        load_verifying_key(str(empty))


def test_the_key_is_read_from_the_configured_path(
    tmp_path: Path, keys: tuple[str, str]
) -> None:
    _, public_pem = keys
    path = tmp_path / "public.pem"
    path.write_text(public_pem)
    assert load_verifying_key(str(path)).startswith("-----BEGIN PUBLIC KEY-----")
