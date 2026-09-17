"""Canonical Reactome user guide URLs for ingestion.

This is the one place in this repo that *fetches* from reactome.org rather than
linking to it, so it is the one worth thinking about. Ten pages, and only when the
userguide bundle is regenerated -- not per request and not per push.

**It fetches beta, not production.** Adam's instruction on 2026-09-17: nothing in
this repo should be making requests to reactome.org. The MCP moved the same day for
the same reason -- this is a dev host, and a rebuild here should not put load on the
public site.

The trade, stated so it is not rediscovered as a surprise: beta's user guide can
describe interface changes that have not reached production, so a bundle built from
beta may explain a UI some readers cannot see yet. That is a content-freshness risk,
not a correctness one -- the guide is a description of the software, and beta is
where this deployment's software comes from. If a released bundle is ever built for
production users, point this at production for that build with the variable below.

Overridable so either target is one environment variable away.
"""

import os

REACTOME_BASE = os.getenv("REACTOME_USERGUIDE_BASE", "https://beta.reactome.org")

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
