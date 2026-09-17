"""Canonical Reactome user guide URLs for ingestion.

This is the one place in this repo that *fetches* from reactome.org rather than
linking to it, so it is the one worth thinking about. Ten pages, and only when the
userguide bundle is regenerated -- not per request and not per push.

**Production, and only because nowhere else serves the guide as HTML.** The
instruction on 2026-09-17 was that this repo should stop making requests to
reactome.org, and the MCP moved to beta that day. This fetch was moved too, and moved
back after measuring what the alternatives actually return:

| source | visible text | what it is |
|---|---|---|
| `reactome.org/userguide` | **2,372 words** | server-rendered Joomla -- the guide |
| `beta.reactome.org/userguide` | 1,245 words | Angular shell; the "text" is inlined CSS |
| `127.0.0.1:4200/userguide` (internal) | 1,237 words | the same Angular app, below the edge |

beta and the internal route are the same application, and it renders the guide in the
browser. A plain HTTP fetch of either returns font declarations, not documentation.
The installed Release95 bundle was built from the Joomla page and carries the same
2,372 words, which is what makes the comparison meaningful.

So a bundle built from beta would contain stylesheets instead of the user guide, and
would pass every structural check on the way -- ten files, right dimensions, non-zero
document count -- while being useless to answer with. That is worse than the load it
would save, which is ten requests a few times a year.

This stops being true the moment the guide is server-rendered somewhere other than
production, or the Angular app exposes the content over an API. Re-measure before
assuming it still holds.
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
