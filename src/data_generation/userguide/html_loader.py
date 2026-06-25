import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SPLIT_CANDIDATES = ("h2", "h3", "h4")
MIN_SECTION_HEADINGS = 2
MAX_CHUNK_CHARS = 4000
MIN_CHUNK_CHARS = 80
CHUNK_OVERLAP = 200

SPAMBOT_PATTERN = re.compile(
    r"This email address is being protected from spambots.*",
    re.IGNORECASE | re.DOTALL,
)
COLLECTIBLE_TAGS = frozenset(
    {
        "p",
        "ul",
        "ol",
        "table",
        "blockquote",
        "pre",
        "div",
        "dl",
        "h3",
        "h4",
        "h5",
        "h6",
    }
)


def choose_split_tag(article_body: Tag) -> str | None:
    """Pick the shallowest heading level with enough sections for this page."""
    for tag in SPLIT_CANDIDATES:
        if len(article_body.find_all(tag)) >= MIN_SECTION_HEADINGS:
            return tag
    return None


def _heading_title(heading: Tag) -> str:
    title = heading.get_text(separator=" ", strip=True)
    if title:
        return title
    image = heading.find("img", alt=True)
    if image is not None:
        alt = image.get("alt", "").strip()
        if alt:
            return alt
    return ""


def split_article_body_into_sections(
    article_body: Tag,
) -> list[tuple[str, int, list[Tag]]]:
    """Split article body into titled sections using adaptive heading boundaries."""
    split_tag = choose_split_tag(article_body)
    if split_tag is None:
        return [("Introduction", 0, _collect_all_content(article_body))]

    headings = article_body.find_all(split_tag)
    sections: list[tuple[str, int, list[Tag]]] = []

    intro_nodes = _collect_intro(article_body, headings[0])
    if intro_nodes:
        sections.append(("Introduction", 0, intro_nodes))

    for index, heading in enumerate(headings):
        title = _heading_title(heading)
        level = int(split_tag[1])
        next_heading = headings[index + 1] if index + 1 < len(headings) else None
        nodes = _collect_between(heading, next_heading, split_tag)
        if not title and not nodes:
            continue
        sections.append((title or "Untitled", level, nodes))

    return sections


def _should_collect_block(element: Tag, collected: list[Tag]) -> bool:
    if element.name not in COLLECTIBLE_TAGS:
        return False
    if not element.get_text(strip=True):
        return False
    for other in collected:
        if element in other.descendants or other in element.descendants:
            return False
    return True


def _collect_intro(article_body: Tag, first_heading: Tag) -> list[Tag]:
    collected: list[Tag] = []
    for element in article_body.descendants:
        if element is first_heading:
            break
        if isinstance(element, Tag) and _should_collect_block(element, collected):
            collected.append(element)
    return collected


def _collect_between(
    start_heading: Tag,
    end_heading: Tag | None,
    split_tag: str,
) -> list[Tag]:
    collected: list[Tag] = []
    for element in start_heading.next_elements:
        if end_heading is not None and element is end_heading:
            break
        if isinstance(element, Tag) and element.name == split_tag:
            break
        if isinstance(element, Tag) and _should_collect_block(element, collected):
            collected.append(element)
    return collected


def _collect_all_content(article_body: Tag) -> list[Tag]:
    collected: list[Tag] = []
    for element in article_body.descendants:
        if isinstance(element, Tag) and _should_collect_block(element, collected):
            collected.append(element)
    return collected


class UserGuideHTMLLoader(BaseLoader):
    """Loads Reactome user guide HTML pages into section-level documents.

    Each document represents one section of a user guide page. The loader picks
    the shallowest heading level (``h2``, ``h3``, or ``h4``) that yields at
    least two sections on that page. Deeper headings remain within their parent
    section. Oversized sections are split for embedding.

    The ``source`` metadata field is set to the canonical page URL. Section
    titles and page titles are included in both metadata and ``page_content``.

    Output Example:
        .. code-block:: txt

            Page: Pathway Browser
            Section: Event Hierarchy

            The order of reactions from top to bottom...
    """

    def __init__(self, html_paths: dict[str, Path]) -> None:
        """
        Args:
            html_paths: Mapping of canonical page URLs to local HTML file paths.
        """
        self.html_paths = html_paths
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_CHUNK_CHARS,
            chunk_overlap=CHUNK_OVERLAP,
        )

    def load(self) -> list[Document]:
        """Load data into document objects."""
        documents: list[Document] = []
        for url, path in self.html_paths.items():
            documents.extend(self._load_page(url, path))
        return documents

    def _load_page(self, url: str, path: Path) -> list[Document]:
        html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "lxml")
        page_title = self._extract_page_title(soup)
        article_body = soup.select_one('[itemprop="articleBody"]')
        if article_body is None:
            raise ValueError(f"No article body found for {url}")

        sections = split_article_body_into_sections(article_body)

        documents: list[Document] = []
        for section_title, section_level, nodes in sections:
            text = self._nodes_to_text(nodes)
            text = SPAMBOT_PATTERN.sub("", text).strip()
            if section_level == 0 and len(text) < MIN_CHUNK_CHARS:
                continue
            if section_level > 0 and len(text) < MIN_CHUNK_CHARS:
                text = f"{section_title}\n\n{text}".strip() if text else section_title

            page_content_prefix = (
                f"URL: {url}\nPage: {page_title}\nSection: {section_title}\n\n"
            )
            chunks = self._splitter.split_text(text)
            for chunk_index, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=page_content_prefix + chunk,
                        metadata={
                            "source": url,
                            "page_title": page_title,
                            "section_title": section_title,
                            "section_level": str(section_level),
                            "chunk_index": str(chunk_index),
                        },
                    )
                )
        return documents

    def _extract_page_title(self, soup: BeautifulSoup) -> str:
        header = soup.select_one(".page-header h2")
        if header:
            title = header.get_text(strip=True)
            if title:
                return title
        if soup.title and soup.title.string:
            return soup.title.string.replace(" - Reactome Pathway Database", "").strip()
        return "Unknown"

    def _nodes_to_text(self, nodes: list[Tag]) -> str:
        parts: list[str] = []
        for node in nodes:
            if node.name == "ul":
                for li in node.find_all("li", recursive=False):
                    item = li.get_text(separator=" ", strip=True)
                    if item:
                        parts.append(f"- {item}")
            elif node.name == "ol":
                for i, li in enumerate(node.find_all("li", recursive=False), start=1):
                    item = li.get_text(separator=" ", strip=True)
                    if item:
                        parts.append(f"{i}. {item}")
            elif node.name == "table":
                text = node.get_text(separator=" ", strip=True)
                if text:
                    parts.append(text)
            else:
                text = node.get_text(separator="\n", strip=True)
                if text:
                    parts.append(text)
        return "\n\n".join(parts)
