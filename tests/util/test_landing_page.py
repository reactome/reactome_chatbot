"""The pages around the chat, and which deployment's brand each one wears.

Rendered directly rather than through `bin/chat-fastapi.py`: importing that mounts
Chainlit, whose configuration is global, and it refuses to start without Turnstile
keys -- see `tests/api/test_chat_requires_human.py` for the lengths that goes to.

What is worth pinning is that the two deployments are the same image. So these
check which `config.yml` changes the page, and that the configuration needing no
change at all is Reactome's: its page has to come out the way it always did, from
the shipped defaults, with nothing for its operators to do.
"""

import logging
from pathlib import Path

import pytest

from agent.profile_names import ProfileName
from util.branding import BRANDING, DEFAULT_BRANDING, Branding, branding_for
from util.config_yml import CONFIG_DEFAULT_YML, Config
from util.landing_page import render_error_page, render_landing_page

REPO_ROOT = Path(__file__).parents[2]

REACTOME_TITLE = "<h1>Meet the React-to-Me AI Chatbot!</h1>"
REACTOME_TAGLINE = (
    "Your new guide to Reactome. Whether you're looking for specific genes and "
    "pathways or just browsing, our AI Chatbot is here to assist you!"
)
REACTOME_LOGO = "https://reactome.org/templates/favourite/images/logo/logo.png"
REACTOME_FAVICON = "https://reactome.org/templates/favourite/favicon.ico"
HELP_EMAIL = "help@reactome.org"


def _landing(branding: Branding, chat_url: str = "https://example.org") -> str:
    return render_landing_page(branding, chat_url)


def test_no_config_at_all_is_still_the_reactome_page() -> None:
    """A checkout with no config.yml, and a config that names no profile."""
    assert branding_for(None) is DEFAULT_BRANDING
    assert branding_for([]) is DEFAULT_BRANDING


def test_the_shipped_default_config_yields_the_reactome_page() -> None:
    """The fallback a Reactome deployment actually runs on.

    Not a fixture written here: this is the file that decides it, so reading the
    shipped one is the only assertion that would notice if it changed.
    """
    config = Config.from_yaml(CONFIG_DEFAULT_YML)
    assert config is not None
    assert branding_for(config.profiles) is DEFAULT_BRANDING


@pytest.mark.parametrize("profile", ["Plant Reactome", ProfileName.Plant_Reactome])
def test_a_profile_and_its_plain_string_agree(profile: str) -> None:
    """config.yml yields plain strings; the typed Config yields enum members.

    Worth a test because `Enum` hashes on the member's name (`Plant_Reactome`)
    while comparing equal to its value (`Plant Reactome`), so a dict keyed by the
    member would miss a string and fall back to Reactome's page -- silently, in
    the one deployment that needs it not to.
    """
    assert branding_for([profile]) is BRANDING["plant reactome"]


def test_profiles_that_disagree_on_a_brand_fall_back_and_say_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One page cannot be the guide to two knowledgebases, so it picks the
    default -- but not quietly."""
    profiles = [ProfileName.React_to_Me, ProfileName.Plant_Reactome]
    with caplog.at_level(logging.WARNING):
        assert branding_for(profiles) is DEFAULT_BRANDING
    assert "do not agree" in caplog.text


def test_every_profile_has_a_branding() -> None:
    """A profile added without one would get Reactome's page in silence."""
    assert set(BRANDING) == {profile.lower() for profile in ProfileName}


def test_the_reactome_page_is_the_page_it_has_always_been() -> None:
    page = _landing(DEFAULT_BRANDING)
    assert REACTOME_TITLE in page
    assert REACTOME_TAGLINE in page
    assert f'href="{REACTOME_FAVICON}"' in page
    assert f'src="{REACTOME_LOGO}"' in page
    assert f'href="mailto:{HELP_EMAIL}"' in page
    assert "Plant Reactome" not in page


def test_the_plant_page_names_plant_reactome() -> None:
    plant = BRANDING["plant reactome"]
    page = _landing(plant)
    assert "<h1>Meet the Plant Reactome AI Chatbot!</h1>" in page
    assert "Your new guide to Plant Reactome." in page
    assert 'alt="Plant Reactome Logo"' in page
    assert plant.home_url in page
    assert plant.logo_url in page
    assert plant.feedback_url in page


def test_the_plant_page_sends_no_one_to_reactome() -> None:
    """The reason this exists: a Plant Reactome reader asking Reactome for help.

    `plantreactome.gramene.org` contains "reactome", so the checks below name the
    Reactome paths rather than the domain.
    """
    page = _landing(BRANDING["plant reactome"])
    assert HELP_EMAIL not in page
    assert "reactome.org/templates" not in page
    assert 'href="https://reactome.org"' not in page


@pytest.mark.parametrize("branding", list(BRANDING.values()))
def test_nothing_is_left_unrendered(branding: Branding) -> None:
    """A placeholder the render does not fill is a page with `$LOGO_URL` on it."""
    assert "$" not in _landing(branding)
    assert "$" not in render_error_page(branding, "Try again", "/chat/guest")


@pytest.mark.parametrize("branding", list(BRANDING.values()))
def test_every_page_offers_both_ways_in_and_feedback(branding: Branding) -> None:
    page = _landing(branding, "https://example.org")
    assert 'href="https://example.org/chat/guest/"' in page
    assert 'href="https://example.org/chat/personal/"' in page
    assert branding.feedback_url in page


def test_an_unset_chat_url_gives_relative_links_not_the_word_none() -> None:
    """CHAINLIT_URL used to interpolate as the literal "None", so every button
    read "None/chat/guest/" and went nowhere."""
    page = _landing(DEFAULT_BRANDING, "")
    assert 'href="/chat/guest/"' in page
    assert "None/chat" not in page


def test_the_error_page_names_the_same_knowledgebase() -> None:
    """The page a failed human check ends on is part of the introduction."""
    reactome = render_error_page(DEFAULT_BRANDING, "CAPTCHA failed", "/chat/guest")
    assert "<h1>CAPTCHA failed</h1>" in reactome
    assert HELP_EMAIL in reactome
    assert 'action="/chat/guest"' in reactome

    plant = render_error_page(BRANDING["plant reactome"], "CAPTCHA failed", "/chat")
    assert HELP_EMAIL not in plant
    assert BRANDING["plant reactome"].support_email in plant


def test_the_landing_page_is_not_duplicated_in_the_app() -> None:
    """The HTML lives in one place, so the two deployments cannot drift."""
    source = (REPO_ROOT / "bin" / "chat-fastapi.py").read_text()
    assert "render_landing_page" in source
    assert "feedback-button" not in source
