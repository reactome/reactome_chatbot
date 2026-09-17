import re
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import requests

USER_AGENT = "ReactomeChatbot/1.0 (+https://github.com/reactome/reactome_chatbot)"

# The thinnest real guide page is review-status at about 400 words; the shell that
# prompted this floor is ~1,240 words of CSS, so a raw character count cannot
# separate them. Stripping tags first is what makes it work: the shell's bulk sits
# inside <style>, so its *visible* text is small.
MIN_PAGE_WORDS = 250


def _visible_word_count(html: str) -> int:
    """Words a reader would see: script and style *contents* removed, not just tags.

    Stripping tags alone is not enough and that is the whole difficulty. The shell
    this guard exists for carries its bulk inside <style>, so tag-stripping leaves
    about 1,240 words of CSS -- more than the thinnest real page has of prose. A
    floor set against that number cannot separate them; removing the blocks first
    takes the shell to near zero and leaves real pages untouched.
    """
    without_blocks = re.sub(
        r"<(script|style)\b[^>]*>.*?</\1>", " ", html, flags=re.S | re.I
    )
    return len(re.sub(r"<[^>]*>", " ", without_blocks).split())


REQUEST_DELAY_SECONDS = 0.5


def url_to_slug(url: str) -> str:
    """Derive a filesystem-safe slug from a user guide URL path."""
    path = urlparse(url).path.strip("/")
    return path.replace("/", "_") if path else "index"


def fetch_userguide_pages(
    urls: Sequence[str],
    cache_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Download user guide HTML pages, using on-disk cache when available.

    Args:
        urls: Canonical user guide URLs to download.
        cache_dir: Directory for cached ``.html`` files.
        force: When ``True``, re-download pages even if cached.

    Returns:
        Mapping of each URL to its local cached HTML path.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    html_paths: dict[str, Path] = {}
    for i, url in enumerate(urls):
        cache_path = cache_dir / f"{url_to_slug(url)}.html"
        if cache_path.exists() and not force:
            html_paths[url] = cache_path
            continue

        response = session.get(url, timeout=60)
        if response.status_code == 403:
            # Non-production hosts serve block-all-automation.conf, which blocks
            # anything self-identifying as automation -- which this deliberately
            # does. A bare 403 reads as a missing page; it is not.
            raise RuntimeError(
                f"403 fetching {url} as User-Agent {USER_AGENT!r}.\n"
                "That host blocks self-identified automation. Note that beta and "
                "the internal Angular app serve the guide as a client-rendered "
                "shell anyway, so fetching them yields stylesheets rather than "
                "documentation -- see urls.py. Production is the only source that "
                "renders it as HTML."
            )
        response.raise_for_status()
        # A 200 is not evidence that a page has content. beta and the internal
        # Angular app both answer 200 for /userguide and return a client-rendered
        # shell whose visible text is inlined CSS -- about 1,240 words of font
        # declarations against production's 2,372 of documentation. A bundle built
        # from that passes every structural check and answers nothing.
        words = _visible_word_count(response.text)
        if words < MIN_PAGE_WORDS:
            raise RuntimeError(
                f"{url} returned {response.status_code} but only {words} words "
                f"of visible text (floor {MIN_PAGE_WORDS}). That is what a "
                "single-page-app shell looks like: it renders in the browser and "
                "a plain fetch gets stylesheets. See urls.py for which hosts "
                "server-render the guide."
            )
        cache_path.write_text(response.text, encoding=response.encoding or "utf-8")
        html_paths[url] = cache_path

        if i < len(urls) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    return html_paths
