"""Utility functions for obsidian-rag: wikilinks, frontmatter, section manipulation."""
from __future__ import annotations

import frontmatter
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

# ── Path helpers ───────────────────────────────────────────────────────────────

def get_safe_file_path(vault_path: str, user_input_path: str) -> Path:
    """Resolve relative path against vault root. Raises on traversal."""
    vault = Path(vault_path).resolve()
    resolved = (vault / user_input_path).resolve()
    if not str(resolved).startswith(str(vault) + str(Path("/"))) and resolved != vault:
        raise ValueError("Security Error: Path traversal detected.")
    return resolved


def list_notes_pattern(subfolder: Optional[str] = None) -> str:
    """Glob pattern for obsidian_list_notes."""
    if subfolder:
        return str(Path(subfolder) / "**" / "*.md")
    return "**/*.md"


# ── Wikilink helpers ──────────────────────────────────────────────────────────

WIKILINK_RE = re.compile(r"\[\[(.*?)(?:\|.*?)?\]\]", re.UNICODE)


def extract_wikilinks(content: str) -> list[str]:
    """Extract all deduplicated wikilink targets from markdown content.
    
    Handles [[Simple]] and [[Link|Alias]] forms.
    Returns a list of unique link targets (without the surrounding [[]]).
    """
    links = WIKILINK_RE.findall(content)
    seen: set[str] = set()
    result = []
    for link in links:
        if link not in seen:
            seen.add(link)
            result.append(link)
    return result


def strip_heading_from_link(link: str) -> str:
    """Strip the #heading fragment from a wikilink target."""
    idx = link.find("#")
    return link if idx == -1 else link[:idx]


# ── Section helpers ────────────────────────────────────────────────────────────

@dataclass
class SectionRange:
    """Represents the character range of a markdown section."""
    heading_start: int
    heading_end: int
    body_start: int
    body_end: int
    level: int


def find_section_range(content: str, heading: str) -> Optional[SectionRange]:
    """Find the range of a section under a heading in markdown content.

    Returns None if heading not found.
    The section body is everything between this heading and the next heading
    of the same or higher level (fewer or equal # symbols).
    """
    escaped = re.escape(heading)
    pattern = re.compile(r"^(#{1,6})\s+" + escaped + r"\s*$", re.MULTILINE)
    match = pattern.search(content)
    if not match:
        return None

    level = len(match.group(1))
    heading_start = match.start()
    heading_end = match.end()
    body_start = heading_end

    # Find next heading at same or higher level
    rest = content[body_start:]
    next_pattern = re.compile(r"^#{1," + str(level) + r"}\s", re.MULTILINE)
    next_match = next_pattern.search(rest)
    body_end = body_start + next_match.start() if next_match else len(content)

    return SectionRange(
        heading_start=heading_start,
        heading_end=heading_end,
        body_start=body_start,
        body_end=body_end,
        level=level,
    )


def replace_section(content: str, range: SectionRange, new_body: str) -> str:
    """Replace the body under a heading, preserving the heading line itself.
    
    Returns the updated file content.
    """
    return content[: range.body_start] + "\n" + new_body + "\n" + content[range.body_end :]


def insert_at_heading(
    content: str,
    heading: str,
    new_content: str,
    position: str = "end",
    range: Optional[SectionRange] = None,
) -> str:
    """Insert content under a heading at beginning or end.

    If heading not found, appends a new ## section.
    """
    if range:
        if position == "beginning":
            return content[: range.body_start] + "\n" + new_content + content[range.body_start :]
        before = content[: range.body_end]
        sep = "\n" if not before.endswith("\n") else ""
        return before + sep + new_content + "\n" + content[range.body_end :]
    return content + f"\n\n## {heading}\n{new_content}"


def replace_in_note(content: str, old_text: str, new_text: str) -> str:
    """Replace the first occurrence of old_text with new_text. Raises if not found."""
    idx = content.find(old_text)
    if idx == -1:
        raise ValueError(f'Text not found: "{old_text}"')
    return content[:idx] + new_text + content[idx + len(old_text) :]


# ── Frontmatter helpers ────────────────────────────────────────────────────────

def parse_frontmatter(content: str) -> tuple[str, dict]:
    """Parse frontmatter from raw markdown content.

    Returns (body, frontmatter_dict).
    """
    post = frontmatter.parse(content)
    if not post:
        return content, {}
    fm_dict = dict(post[0]) if post[0] else {}
    return post[1], fm_dict


def apply_frontmatter_update(
    content: str,
    updates: Optional[dict] = None,
    key: Optional[str] = None,
    value: Optional[str] = None,
) -> str:
    """Update YAML frontmatter of a note.

    Supports two modes:
    - Batch: updates = {"key": value, ...}
    - Single: key + value

    Values that look like JSON are parsed automatically.
    """
    post = frontmatter.parse(content)
    if post and post[0]:
        metadata = dict(post[0])
        body = post[1]
    else:
        metadata = {}
        body = content

    if updates:
        for k, v in updates.items():
            if k in ("__proto__", "constructor", "prototype"):
                continue
            # Try to parse as JSON for complex types
            try:
                metadata[k] = json.loads(v) if isinstance(v, str) else v
            except (json.JSONDecodeError, TypeError):
                metadata[k] = v
    elif key is not None:
        if value is not None:
            try:
                val = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                val = value
            metadata[key] = val
        else:
            del metadata[key]

    return frontmatter.dumps(frontmatter.Post(body, **metadata))


import json  # noqa: E402 — used in apply_frontmatter_update
