"""Canonical Reactome user guide URLs for ingestion.

This is the one place in this repo that *fetches* from reactome.org rather than
linking to it, so it is the one worth thinking about. Ten pages, and only when the
userguide bundle is regenerated -- not per request and not per push.

**It stays on production on purpose, and that is not an oversight.** The MCP was
moved to beta.reactome.org on 2026-09-17 because a test gate should not lean on the
service it protects, and the two hosts answer identically there: same release, same
species count. That reasoning does not carry here. The user guide documents the site
people are actually using, and beta's guide can describe interface changes that have
not shipped. A bundle built from beta would confidently explain a UI the reader
cannot see.

Overridable for testing, so a rebuild can be pointed elsewhere without editing this
file -- but the default is deliberate.
"""

import os

REACTOME_BASE = os.getenv("REACTOME_USERGUIDE_BASE", "https://reactome.org")

USER_GUIDE_URLS: tuple[str, ...] = (
    f"{REACTOME_BASE}/userguide",
    f"{REACTOME_BASE}/userguide/pathway-browser",
    f"{REACTOME_BASE}/userguide/searching",
    f"{REACTOME_BASE}/userguide/details-panel",
    f"{REACTOME_BASE}/userguide/analysis",
    f"{REACTOME_BASE}/userguide/analysis/gsa",
    f"{REACTOME_BASE}/userguide/diseases",
    f"{REACTOME_BASE}/userguide/cytomics",
    f"{REACTOME_BASE}/userguide/review-status",
    f"{REACTOME_BASE}/userguide/reactome-fiviz",
)
