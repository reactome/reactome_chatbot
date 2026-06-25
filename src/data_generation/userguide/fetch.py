import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import requests

USER_AGENT = "ReactomeChatbot/1.0 (+https://github.com/reactome/reactome_chatbot)"
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
        response.raise_for_status()
        cache_path.write_text(response.text, encoding=response.encoding or "utf-8")
        html_paths[url] = cache_path

        if i < len(urls) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    return html_paths
