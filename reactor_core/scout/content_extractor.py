"""
Content Extractor for Safe Scout.

Provides:
- Clean text extraction from HTML
- Code block preservation
- Heading structure extraction
- Documentation-specific parsing
- Boilerplate removal
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Type of extracted content."""
    DOCUMENTATION = "documentation"
    TUTORIAL = "tutorial"
    API_REFERENCE = "api_reference"
    BLOG_POST = "blog_post"
    FORUM_POST = "forum_post"
    CODE_REPOSITORY = "code_repository"
    UNKNOWN = "unknown"


@dataclass
class CodeBlock:
    """Extracted code block."""
    code: str
    language: str = ""
    title: str = ""
    line_start: int = 0
    context: str = ""  # Surrounding text


@dataclass
class Section:
    """Document section."""
    heading: str
    level: int  # 1-6 for h1-h6
    content: str
    code_blocks: List[CodeBlock] = field(default_factory=list)
    subsections: List["Section"] = field(default_factory=list)


@dataclass
class ExtractedContent:
    """Result of content extraction."""
    url: str
    title: str
    content_type: ContentType
    description: str = ""

    # Main content
    text_content: str = ""
    markdown_content: str = ""

    # Structured elements
    sections: List[Section] = field(default_factory=list)
    code_blocks: List[CodeBlock] = field(default_factory=list)
    links: List[Tuple[str, str]] = field(default_factory=list)  # (text, href)

    # Metadata
    author: str = ""
    published_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    word_count: int = 0
    reading_time_minutes: float = 0.0

    # Quality metrics
    quality_score: float = 0.0
    has_code_examples: bool = False
    has_structured_content: bool = False

    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content_type": self.content_type.value,
            "description": self.description,
            "text_content_length": len(self.text_content),
            "markdown_content_length": len(self.markdown_content),
            "section_count": len(self.sections),
            "code_block_count": len(self.code_blocks),
            "link_count": len(self.links),
            "author": self.author,
            "word_count": self.word_count,
            "reading_time_minutes": self.reading_time_minutes,
            "quality_score": self.quality_score,
            "has_code_examples": self.has_code_examples,
            "has_structured_content": self.has_structured_content,
            "extracted_at": self.extracted_at.isoformat(),
        }


@dataclass
class ExtractorConfig:
    """Configuration for content extractor."""
    # Content filtering
    min_content_length: int = 100
    max_content_length: int = 500000  # 500KB text limit

    # Code extraction
    preserve_code_blocks: bool = True
    detect_code_language: bool = True

    # Boilerplate removal
    remove_navigation: bool = True
    remove_footer: bool = True
    remove_sidebar: bool = True
    remove_ads: bool = True
    remove_comments_section: bool = True

    # Link handling
    extract_internal_links: bool = True
    resolve_relative_links: bool = True

    # Reading time calculation
    words_per_minute: int = 200


# Boilerplate element selectors
BOILERPLATE_SELECTORS = [
    # Navigation
    "nav", "header", ".nav", ".navbar", ".navigation",
    "#nav", "#navbar", "#navigation", ".menu", "#menu",
    ".top-bar", ".topbar", ".header-nav",

    # Footer
    "footer", ".footer", "#footer", ".site-footer",
    ".page-footer", ".bottom-bar",

    # Sidebars
    "aside", ".sidebar", "#sidebar", ".side-nav",
    ".widget", ".widgets", ".related-posts",

    # Ads and promotions
    ".ad", ".ads", ".advertisement", ".sponsored",
    ".promo", ".promotion", ".banner", ".newsletter-signup",

    # Comments
    ".comments", "#comments", ".comment-section",
    ".disqus", "#disqus_thread",

    # Social sharing
    ".share", ".social", ".social-share", ".sharing",

    # Cookie banners
    ".cookie", ".cookie-banner", ".cookie-consent",
    ".gdpr", ".privacy-notice",

    # Popups
    ".modal", ".popup", ".overlay", ".lightbox",
]

# Content area selectors (prioritized)
CONTENT_SELECTORS = [
    # Documentation
    "article", "main", ".content", "#content",
    ".main-content", "#main-content", ".article-content",
    ".doc-content", ".documentation", ".docs",
    ".markdown-body", ".prose", ".post-content",
    ".entry-content", ".page-content", "[role='main']",

    # Specific platforms
    ".rst-content",  # ReadTheDocs
    ".md-content",   # MkDocs
    ".document",     # Sphinx
]


class ContentExtractor:
    """
    Extracts clean, structured content from HTML.

    Optimized for technical documentation with:
    - Code block preservation
    - Heading structure extraction
    - Boilerplate removal
    - Markdown output generation
    """

    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
    ):
        self.config = config or ExtractorConfig()

    async def extract(
        self,
        url: str,
        html: str,
        raw_text: Optional[str] = None,
    ) -> ExtractedContent:
        """
        Extract content from HTML.

        Args:
            url: Source URL
            html: Raw HTML content
            raw_text: Optional pre-extracted text

        Returns:
            ExtractedContent with structured data
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "BeautifulSoup required. Install with: pip install beautifulsoup4"
            )

        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = self._extract_title(soup)

        # Extract description
        description = self._extract_description(soup)

        # Remove boilerplate
        self._remove_boilerplate(soup)

        # Find main content area
        main_content = self._find_main_content(soup)

        # Extract code blocks before text processing
        code_blocks = self._extract_code_blocks(main_content)

        # Extract sections with headings
        sections = self._extract_sections(main_content)

        # Extract text content
        text_content = self._extract_text(main_content)

        # Generate markdown
        markdown_content = self._generate_markdown(
            title, description, sections, code_blocks
        )

        # Extract links
        links = self._extract_links(main_content, url)

        # Extract metadata
        author, published_date, last_modified = self._extract_metadata(soup)

        # Calculate metrics
        word_count = len(text_content.split())
        reading_time = word_count / self.config.words_per_minute

        # Determine content type
        content_type = self._detect_content_type(url, soup)

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            text_content, code_blocks, sections
        )

        return ExtractedContent(
            url=url,
            title=title,
            content_type=content_type,
            description=description,
            text_content=text_content,
            markdown_content=markdown_content,
            sections=sections,
            code_blocks=code_blocks,
            links=links,
            author=author,
            published_date=published_date,
            last_modified=last_modified,
            word_count=word_count,
            reading_time_minutes=reading_time,
            quality_score=quality_score,
            has_code_examples=len(code_blocks) > 0,
            has_structured_content=len(sections) > 0,
        )

    def _extract_title(self, soup) -> str:
        """Extract page title."""
        # Try og:title first
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        # Try title tag
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()
            # Remove site name suffix
            title = re.sub(r'\s*[|\-–—]\s*[^|\-–—]+$', '', title)
            return title

        # Try h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text().strip()

        return ""

    def _extract_description(self, soup) -> str:
        """Extract page description."""
        # Try og:description
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            return og_desc["content"].strip()

        # Try meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return meta_desc["content"].strip()

        return ""

    def _remove_boilerplate(self, soup) -> None:
        """Remove boilerplate elements."""
        if not self.config.remove_navigation:
            # Keep selective boilerplate
            selectors = BOILERPLATE_SELECTORS[:]
            selectors = [s for s in selectors if not any(
                nav in s for nav in ["nav", "menu", "header"]
            )]
        else:
            selectors = BOILERPLATE_SELECTORS

        for selector in selectors:
            for element in soup.select(selector):
                element.decompose()

        # Remove scripts and styles
        for tag in ["script", "style", "noscript", "svg", "iframe"]:
            for element in soup.find_all(tag):
                element.decompose()

        # Remove hidden elements
        for element in soup.find_all(attrs={"hidden": True}):
            element.decompose()

        for element in soup.find_all(style=re.compile(r"display:\s*none")):
            element.decompose()

    def _find_main_content(self, soup):
        """Find the main content area."""
        for selector in CONTENT_SELECTORS:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > self.config.min_content_length:
                return content

        # Fallback to body
        return soup.find("body") or soup

    def _extract_code_blocks(self, content) -> List[CodeBlock]:
        """Extract code blocks from content."""
        if not self.config.preserve_code_blocks:
            return []

        code_blocks = []

        # Find pre/code blocks
        for pre in content.find_all("pre"):
            code_elem = pre.find("code")
            code_text = (code_elem or pre).get_text()

            # Detect language
            language = ""
            if code_elem:
                classes = code_elem.get("class", [])
                for cls in classes:
                    if cls.startswith(("language-", "lang-", "hljs-")):
                        language = cls.split("-", 1)[1]
                        break

            # Get context (preceding paragraph)
            context = ""
            prev_elem = pre.find_previous_sibling(["p", "div"])
            if prev_elem:
                context = prev_elem.get_text(strip=True)[:200]

            if code_text.strip():
                code_blocks.append(CodeBlock(
                    code=code_text.strip(),
                    language=language,
                    context=context,
                ))

        # Find inline code (for smaller snippets)
        for code in content.find_all("code"):
            # Skip if inside pre
            if code.find_parent("pre"):
                continue

            code_text = code.get_text()
            if len(code_text) > 20 and "\n" in code_text:
                code_blocks.append(CodeBlock(
                    code=code_text.strip(),
                    language="",
                ))

        return code_blocks

    def _extract_sections(self, content) -> List[Section]:
        """Extract document sections based on headings."""
        sections = []
        current_sections = {i: None for i in range(1, 7)}

        for heading in content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            level = int(heading.name[1])
            heading_text = heading.get_text(strip=True)

            # Get content until next heading
            section_content = []
            section_code_blocks = []

            for sibling in heading.find_next_siblings():
                if sibling.name and sibling.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    break

                if sibling.name == "pre":
                    code_elem = sibling.find("code")
                    code_text = (code_elem or sibling).get_text().strip()
                    if code_text:
                        section_code_blocks.append(CodeBlock(code=code_text))
                else:
                    text = sibling.get_text(strip=True)
                    if text:
                        section_content.append(text)

            section = Section(
                heading=heading_text,
                level=level,
                content="\n\n".join(section_content),
                code_blocks=section_code_blocks,
            )

            # Build hierarchy
            if level == 1:
                sections.append(section)
                current_sections[1] = section
            elif current_sections.get(level - 1):
                current_sections[level - 1].subsections.append(section)
                current_sections[level] = section
            else:
                sections.append(section)
                current_sections[level] = section

        return sections

    def _extract_text(self, content) -> str:
        """Extract clean text content."""
        # Get text
        text = content.get_text(separator="\n")

        # Clean up
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                if cleaned_lines and cleaned_lines[-1]:
                    cleaned_lines.append("")
                continue

            # Skip very short lines (likely noise)
            if len(line) < 3 and not line.isdigit():
                continue

            cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)

        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Enforce max length
        if len(text) > self.config.max_content_length:
            text = text[:self.config.max_content_length] + "..."

        return text.strip()

    def _generate_markdown(
        self,
        title: str,
        description: str,
        sections: List[Section],
        code_blocks: List[CodeBlock],
    ) -> str:
        """Generate markdown from extracted content."""
        parts = []

        # Title
        if title:
            parts.append(f"# {title}\n")

        # Description
        if description:
            parts.append(f"> {description}\n")

        # Sections
        for section in sections:
            parts.append(self._section_to_markdown(section))

        # Standalone code blocks
        if code_blocks and not sections:
            parts.append("\n## Code Examples\n")
            for block in code_blocks:
                lang = block.language or ""
                parts.append(f"```{lang}\n{block.code}\n```\n")

        return "\n".join(parts)

    def _section_to_markdown(self, section: Section, depth: int = 0) -> str:
        """Convert section to markdown."""
        parts = []

        # Heading
        heading_level = min(section.level + depth, 6)
        parts.append(f"{'#' * heading_level} {section.heading}\n")

        # Content
        if section.content:
            parts.append(section.content + "\n")

        # Code blocks
        for block in section.code_blocks:
            lang = block.language or ""
            parts.append(f"```{lang}\n{block.code}\n```\n")

        # Subsections
        for subsection in section.subsections:
            parts.append(self._section_to_markdown(subsection, depth))

        return "\n".join(parts)

    def _extract_links(
        self,
        content,
        base_url: str,
    ) -> List[Tuple[str, str]]:
        """Extract links from content."""
        from urllib.parse import urljoin, urlparse

        links = []
        seen_hrefs = set()

        for a in content.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)

            # Skip empty or anchor-only links
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue

            # Resolve relative URLs
            if self.config.resolve_relative_links:
                href = urljoin(base_url, href)

            # Skip duplicates
            if href in seen_hrefs:
                continue
            seen_hrefs.add(href)

            # Filter internal links only if configured
            if self.config.extract_internal_links:
                base_domain = urlparse(base_url).netloc
                link_domain = urlparse(href).netloc
                if link_domain and link_domain != base_domain:
                    continue

            links.append((text, href))

        return links

    def _extract_metadata(self, soup) -> Tuple[str, Optional[datetime], Optional[datetime]]:
        """Extract author and date metadata."""
        author = ""
        published_date = None
        last_modified = None

        # Author
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta:
            author = author_meta.get("content", "")

        # Published date
        date_meta = soup.find("meta", property="article:published_time")
        if date_meta:
            try:
                published_date = datetime.fromisoformat(
                    date_meta["content"].replace("Z", "+00:00")
                )
            except Exception:
                pass

        # Last modified
        modified_meta = soup.find("meta", property="article:modified_time")
        if modified_meta:
            try:
                last_modified = datetime.fromisoformat(
                    modified_meta["content"].replace("Z", "+00:00")
                )
            except Exception:
                pass

        return author, published_date, last_modified

    def _detect_content_type(self, url: str, soup) -> ContentType:
        """Detect type of content."""
        url_lower = url.lower()

        if "/docs/" in url_lower or "documentation" in url_lower:
            return ContentType.DOCUMENTATION
        if "/tutorial" in url_lower or "/learn" in url_lower:
            return ContentType.TUTORIAL
        if "/api" in url_lower or "/reference" in url_lower:
            return ContentType.API_REFERENCE
        if "/blog/" in url_lower or "/posts/" in url_lower:
            return ContentType.BLOG_POST
        if "stackoverflow.com" in url_lower or "github.com/discussions" in url_lower:
            return ContentType.FORUM_POST
        if "github.com" in url_lower:
            return ContentType.CODE_REPOSITORY

        return ContentType.UNKNOWN

    def _calculate_quality_score(
        self,
        text: str,
        code_blocks: List[CodeBlock],
        sections: List[Section],
    ) -> float:
        """Calculate content quality score."""
        score = 0.5  # Base score

        # Length bonus
        word_count = len(text.split())
        if word_count > 200:
            score += 0.1
        if word_count > 500:
            score += 0.1

        # Code examples bonus
        if code_blocks:
            score += 0.1
            if len(code_blocks) >= 3:
                score += 0.05

        # Structured content bonus
        if sections:
            score += 0.1
            if len(sections) >= 3:
                score += 0.05

        # Technical content indicators
        technical_patterns = [
            r'\bfunction\b',
            r'\bclass\b',
            r'\bimport\b',
            r'\breturn\b',
            r'\bAPI\b',
            r'\bHTTP\b',
            r'\bJSON\b',
        ]

        tech_count = sum(
            1 for p in technical_patterns
            if re.search(p, text, re.IGNORECASE)
        )
        score += min(0.1, tech_count * 0.02)

        return min(1.0, score)


# Convenience exports
__all__ = [
    "ContentExtractor",
    "ExtractorConfig",
    "ExtractedContent",
    "ContentType",
    "Section",
    "CodeBlock",
]
