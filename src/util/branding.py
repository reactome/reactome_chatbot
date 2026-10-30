"""Which knowledgebase a deployment is, and how it should introduce itself.

The landing page and the pages a failed human check ends on had Reactome's name,
Reactome's logo and Reactome's support address written into them. The Plant
Reactome deployment is the same image with its own `config.yml` bind-mounted over
the default (see `compose.yaml`), so it was introducing Plant Reactome's readers
to a human knowledgebase -- the same defect `chainlit.plantreactome.md` exists to
fix on the welcome screen, and `tests/test_welcome_pages.py` records how that one
used to be backwards.

The switch is `profiles:` in config.yml. It already decides which graph answers
and which chat profiles are offered, so reusing it means there is one place where
a deployment says who it is; a second switch would be a second thing that can
disagree with the first.

Reactome's branding is the default *in code*, not something its deployment has to
ask for: a checkout with no config.yml falls back to config_default.yml, which
names React-to-Me, and an empty profile list lands here too. Reactome changes
nothing to keep what it has.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from agent.profile_names import ProfileName
from util.logging import logging


@dataclass(frozen=True)
class Branding:
    """What one knowledgebase calls itself on the pages we serve."""

    #: In prose: "Your new guide to Reactome" / "... to Plant Reactome".
    knowledgebase: str
    #: Used in the page title. Reactome's chatbot has its own name; the Plant
    #: Reactome one is named for the knowledgebase, which is also the name of
    #: its chat profile and of the `{chat_profile}` in its welcome message.
    chatbot_name: str
    #: The knowledgebase's own site, which the logo links to.
    home_url: str
    logo_url: str
    favicon_url: str
    #: The Feedback button's target. A mailto for Reactome; Plant Reactome
    #: publishes no address, only a form (their site's Contact Us page).
    feedback_url: str
    #: Shown as text, not as a link, on the error pages -- so it has to read
    #: well on its own. Plant Reactome's is Gramene's published contact
    #: address, since Plant Reactome is a Gramene database. Worth confirming
    #: with them before this page goes public.
    support_email: str


REACTOME = Branding(
    knowledgebase="Reactome",
    chatbot_name="React-to-Me",
    home_url="https://reactome.org",
    logo_url="https://reactome.org/templates/favourite/images/logo/logo.png",
    favicon_url="https://reactome.org/templates/favourite/favicon.ico",
    feedback_url="mailto:help@reactome.org",
    support_email="help@reactome.org",
)

PLANT_REACTOME = Branding(
    knowledgebase="Plant Reactome",
    chatbot_name="Plant Reactome",
    home_url="https://plantreactome.gramene.org",
    logo_url="https://plantreactome.gramene.org/templates/favourite/images/logo/logo.png",
    # Their site serves no icon at the usual /favicon.ico path; the logo scales
    # down to one, and a browser asks for an image either way.
    favicon_url="https://plantreactome.gramene.org/templates/favourite/images/logo/logo.png",
    feedback_url=(
        "https://plantreactome.gramene.org/index.php"
        "?Itemid=446&id=1&lang=en&option=com_contact&view=contact"
    ),
    support_email="gramene@gramene.org",
)

# Keyed by the lower-cased name, as `agent.profiles.CHAT_PROFILES` is. Keying by
# the ProfileName member itself would not survive a plain string: `Enum` hashes
# on the member's *name* (`React_to_Me`) while comparing equal to its value
# (`React-to-Me`), so a string key would miss and land on the default.
BRANDING: dict[str, Branding] = {
    ProfileName.React_to_Me.lower(): REACTOME,
    # Reactome's second profile answers from Reactome and UniProt, and is served
    # from Reactome's site, so it is Reactome's page.
    ProfileName.Cross_Database_Prototype.lower(): REACTOME,
    ProfileName.Plant_Reactome.lower(): PLANT_REACTOME,
}

#: What a deployment that has not said otherwise gets.
DEFAULT_BRANDING = REACTOME


def branding_for(profiles: Sequence[str] | None) -> Branding:
    """The branding for a deployment's configured profiles.

    Unanimous, or the default. A deployment configured for both knowledgebases at
    once is the case this cannot answer -- one page cannot be the guide to both --
    so it says which one it served rather than picking one silently.
    """
    if not profiles:
        return DEFAULT_BRANDING

    chosen = {
        BRANDING.get(str(profile).lower(), DEFAULT_BRANDING) for profile in profiles
    }
    if len(chosen) > 1:
        logging.warning(
            "The configured profiles %s do not agree on one brand; serving the %s "
            "landing page. A deployment that offers both knowledgebases needs a "
            "landing page that names both.",
            [str(profile) for profile in profiles],
            DEFAULT_BRANDING.knowledgebase,
        )
        return DEFAULT_BRANDING
    return chosen.pop()
