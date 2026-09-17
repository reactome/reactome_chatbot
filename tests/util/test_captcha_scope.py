"""Which paths the captcha guards, pinned before a new route is added beside it.

Spec 010 adds an answer endpoint under the same app. The captcha middleware
currently intercepts everything under CHAINLIT_URI, so the new route would be
redirected to a captcha page unless it is exempted -- and exempting it is a
change to an authentication boundary, which should be deliberate rather than
incidental.

These assert the behaviour as it was before the rule was extracted from
bin/chat-fastapi.py, so the extraction is provably behaviour-preserving.

One case is not hypothetical: `captcha_configured` must come from the same place
the signing secret does. The middleware read `os.environ` directly, so a
deployment mounting CLOUDFLARE_SECRET_KEY as a Docker secret -- the mechanism
get_secret documents as preferred -- had the captcha silently switched off.
"""

import pytest

from util.captcha_scope import is_captcha_exempt

URI = "/chat/guest"


def _exempt(path: str, **kw: object) -> bool:
    options: dict[str, object] = {
        "chainlit_uri": URI,
        "captcha_configured": True,
    }
    options.update(kw)
    return is_captcha_exempt(path, **options)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "path",
    [
        "/chat/",
        f"{URI}/verify_captcha",
        f"{URI}/verify_captcha_page",
        f"{URI}/static",
        "/static/app.js",
    ],
)
def test_the_captcha_does_not_guard_its_own_pages(path: str) -> None:
    assert _exempt(path)


@pytest.mark.parametrize("path", [URI, f"{URI}/", f"{URI}/anything"])
def test_the_chat_app_itself_is_guarded(path: str) -> None:
    assert not _exempt(path)


def test_paths_outside_the_chat_app_are_not_ours_to_guard() -> None:
    assert _exempt("/content/detail/R-HSA-109581")
    assert _exempt("/")


def test_no_captcha_configured_means_nothing_is_guarded() -> None:
    """Beta runs this way: the Turnstile site key is bound to reactome.org."""
    assert _exempt(f"{URI}/anything", captcha_configured=False)


def test_a_new_route_is_guarded_until_it_is_explicitly_exempted() -> None:
    """The point of pinning this: spec 010's endpoint needs a deliberate change.

    Without the exemption it would be redirected to a captcha page rather than
    verifying its own caller.
    """
    assert not _exempt(f"{URI}/api/answer")
    assert _exempt(f"{URI}/api/answer", extra_prefixes=[f"{URI}/api/"])


def test_an_unset_chainlit_uri_guards_everything_but_the_landing_page() -> None:
    """Characterization, and not what I expected when writing it.

    With CHAINLIT_URI unset the rule exempts only the explicit list and
    /static, so every other path is guarded -- and the middleware would then
    redirect to f"{CHAINLIT_URI}/verify_captcha_page", rendering as
    "None/verify_captcha_page".

    BUG: that redirect target is nonsense. It is only reachable with
    CHAINLIT_URI unset, which no deployment does, so it is pinned rather than
    fixed -- changing it should be deliberate, with this test moving alongside.
    """
    assert _exempt("/chat/", chainlit_uri=None)
    assert not _exempt("/anything", chainlit_uri=None)
