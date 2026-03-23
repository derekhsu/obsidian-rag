#!/usr/bin/env python3
"""
obsidian-rag — Obsidian RAG CLI (OpenClaw Skill)

A command-line interface for Obsidian vault management and semantic search.

Usage:
    python main.py <command> [options]

Commands:
    list_notes       List markdown notes in vault
    read_note       Read a note's content
    create_note     Create a new note
    append_note     Append text to a note
    search_notes    Full-text search across vault
    get_backlinks   Find notes linking to a target
    get_links       Extract wikilinks from a note
    get_broken_links  Find broken wikilinks
    move_note       Move or rename a note
    update_frontmatter  Update YAML frontmatter
    replace_section  Replace content under a heading
    replace_in_note  Replace first occurrence of text
    get_daily_note   Get or create today's daily note
    append_daily_log  Append to daily note under heading
    rag_index       Index vault for semantic search
    rag_query       Semantic search on indexed vault

Environment:
    OBSIDIAN_VAULT_PATH   Path to the Obsidian vault (or use set_vault)
"""
from __future__ import annotations

import asyncio
import glob as _glob
import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from obsidian_rag.config import config
from obsidian_rag.utils import (
    apply_frontmatter_update,
    extract_wikilinks,
    find_section_range,
    get_safe_file_path,
    insert_at_heading,
    list_notes_pattern,
    parse_frontmatter,
    replace_in_note,
    replace_section,
    strip_heading_from_link,
)
from obsidian_rag.vector_store import VaultIndexer

# ── Shared state ───────────────────────────────────────────────────────────────

_indexer = VaultIndexer()

# ── Vault path helpers ─────────────────────────────────────────────────────────

def vault_path_from_args(vault_path: Optional[str]) -> Path:
    """Resolve vault path: CLI arg > config > env."""
    vp = vault_path or config.vault_path
    if not vp:
        raise click.ClickException(
            "Vault path is not set. "
            "Run: obsidian-rag set_vault /path/to/vault\n"
            "Or set OBSIDIAN_VAULT_PATH env variable."
        )
    return Path(vp).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: set_vault
# ─────────────────────────────────────────────────────────────────────────────

@click.command("set_vault")
@click.argument("path")
def set_vault(path: str):
    """Set the default Obsidian vault path (saved to config)."""
    p = Path(path).resolve()
    if not p.is_dir():
        raise click.ClickException(f"Not a directory: {path}")
    config.set_vault(str(p))
    click.echo(f"Vault path set to: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: list_notes
# ─────────────────────────────────────────────────────────────────────────────

@click.command("list_notes")
@click.option("--vault-path", help="Vault path override")
@click.option("--subfolder", help="Subfolder to list")
@click.option("--limit", default=200, help="Maximum number of files (default: 200)")
def list_notes(vault_path: Optional[str], subfolder: Optional[str], limit: int):
    """List markdown notes in the vault (or a subfolder)."""
    vp = vault_path_from_args(vault_path)
    pattern = list_notes_pattern(subfolder)
    files = sorted(Path(vp).glob(pattern)) if subfolder else sorted(Path(vp).rglob("*.md"))
    if not files:
        click.echo("No notes found.")
        return
    for f in files:
        click.echo(f)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: read_note
# ─────────────────────────────────────────────────────────────────────────────

@click.command("read_note")
@click.argument("file_path")
@click.option("--vault-path", help="Vault path override")
def read_note(file_path: str, vault_path: Optional[str]):
    """Read the full content of a note."""
    vp = vault_path_from_args(vault_path)
    fp = get_safe_file_path(str(vp), file_path)
    if not fp.exists():
        raise click.ClickException(f"File not found: {file_path}")
    click.echo(fp.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: create_note
# ─────────────────────────────────────────────────────────────────────────────

@click.command("create_note")
@click.argument("file_path")
@click.option("--content", default="", help="Initial note content")
@click.option("--vault-path", help="Vault path override")
def create_note(file_path: str, content: str, vault_path: Optional[str]):
    """Create a new note with the given content."""
    vp = vault_path_from_args(vault_path)
    fp = get_safe_file_path(str(vp), file_path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8")
    click.echo(f"Created note: {file_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5: append_note
# ─────────────────────────────────────────────────────────────────────────────

@click.command("append_note")
@click.argument("file_path")
@click.option("--content", required=True, help="Text to append")
@click.option("--vault-path", help="Vault path override")
def append_note(file_path: str, content: str, vault_path: Optional[str]):
    """Append text to the end of an existing note."""
    vp = vault_path_from_args(vault_path)
    fp = get_safe_file_path(str(vp), file_path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("a", encoding="utf-8") as f:
        f.write("\n" + content)
    click.echo(f"Appended to note: {file_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 6: search_notes
# ─────────────────────────────────────────────────────────────────────────────

@click.command("search_notes")
@click.argument("query")
@click.option("--vault-path", help="Vault path override")
@click.option("--limit", default=20, help="Max results (default: 20)")
def search_notes(query: str, vault_path: Optional[str], limit: int):
    """Full-text search across vault notes (filename + content match)."""
    vp = vault_path_from_args(vault_path)
    q = query.lower()
    files = sorted(Path(vp).rglob("*.md"))
    matches = []
    for f in files:
        if q in Path(f).name.lower():
            matches.append(f"{f}  (filename match)")
            if len(matches) >= limit:
                break
    for f in files:
        if len(matches) >= limit:
            break
        if q not in f.name.lower():
            try:
                content = f.read_text(encoding="utf-8")
                if q in content.lower():
                    matches.append(str(f))
            except Exception:
                pass
    if not matches:
        click.echo(f"No results for: {query}")
    else:
        for m in matches:
            click.echo(m)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 7: get_backlinks
# ─────────────────────────────────────────────────────────────────────────────

@click.command("get_backlinks")
@click.argument("file_name")
@click.option("--vault-path", help="Vault path override")
def get_backlinks(file_name: str, vault_path: Optional[str]):
    """Find all notes that link to a specific note."""
    import re

    vp = vault_path_from_args(vault_path)
    # Strip .md extension
    target = Path(file_name).stem
    link_re = re.compile(
        r"\[\[" + re.escape(target) + r"(?:[\\]|\||#|\])",
        re.IGNORECASE,
    )
    files = sorted(vp.rglob("*.md"))
    backlinks = []
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
            if link_re.search(content):
                backlinks.append(str(f))
        except Exception:
            pass
    if not backlinks:
        click.echo(f"No backlinks found for: [[{target}]]")
    else:
        click.echo(f"Found {len(backlinks)} backlink(s) for [[{target}]]:")
        for b in backlinks:
            click.echo(f"  - {b}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 8: get_links
# ─────────────────────────────────────────────────────────────────────────────

@click.command("get_links")
@click.argument("file_path")
@click.option("--vault-path", help="Vault path override")
def get_links(file_path: str, vault_path: Optional[str]):
    """Extract all outgoing wikilinks from a note."""
    vp = vault_path_from_args(vault_path)
    fp = get_safe_file_path(str(vp), file_path)
    content = fp.read_text(encoding="utf-8")
    links = extract_wikilinks(content)
    if not links:
        click.echo("No wikilinks found.")
    else:
        for link in links:
            click.echo(link)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 9: get_broken_links
# ─────────────────────────────────────────────────────────────────────────────

@click.command("get_broken_links")
@click.option("--vault-path", help="Vault path override")
@click.option("--subfolder", help="Limit scan to subfolder")
def get_broken_links(vault_path: Optional[str], subfolder: Optional[str]):
    """Find all wikilinks pointing to non-existent notes."""
    vp = vault_path_from_args(vault_path)
    all_files = sorted(vp.rglob("*.md"))
    files = all_files if not subfolder else [f for f in all_files if str(f).startswith(str(vp / subfolder))]
    name_set = {f.stem.lower() for f in all_files}

    target_map: dict[str, list[str]] = {}
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
        except Exception:
            continue
        for link in extract_wikilinks(content):
            target = strip_heading_from_link(link).strip()
            if not target:
                continue
            refs = target_map.setdefault(target.lower(), [])
            if str(f) not in refs:
                refs.append(str(f))

    broken = [
        {"target": t, "refs": refs}
        for t, refs in target_map.items()
        if t not in name_set
    ]

    if not broken:
        click.echo("No broken links found.")
    else:
        click.echo(f"Found {len(broken)} broken link(s):")
        for entry in broken:
            click.echo(f"  [[{entry['target']}]] — in: {', '.join(entry['refs'])}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 10: move_note
# ─────────────────────────────────────────────────────────────────────────────

@click.command("move_note")
@click.argument("source_path")
@click.argument("dest_path")
@click.option("--vault-path", help="Vault path override")
def move_note(source_path: str, dest_path: str, vault_path: Optional[str]):
    """Move or rename a note."""
    vp = vault_path_from_args(vault_path)
    src = get_safe_file_path(str(vp), source_path)
    dst = get_safe_file_path(str(vp), dest_path)
    if not src.exists():
        raise click.ClickException(f"Source not found: {source_path}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    click.echo(f"Moved {source_path} -> {dest_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 11: update_frontmatter
# ─────────────────────────────────────────────────────────────────────────────

@click.command("update_frontmatter")
@click.argument("file_path")
@click.option("--key", help="Frontmatter key to set (single-key mode)")
@click.option("--value", help="Value for the key")
@click.option("--updates", help="JSON object for batch update")
@click.option("--vault-path", help="Vault path override")
def update_frontmatter(
    file_path: str,
    key: Optional[str],
    value: Optional[str],
    updates: Optional[str],
    vault_path: Optional[str],
):
    """Update YAML frontmatter fields of a note."""
    vp = vault_path_from_args(vault_path)
    fp = get_safe_file_path(str(vp), file_path)
    if not fp.exists():
        raise click.ClickException(f"File not found: {file_path}")

    content = fp.read_text(encoding="utf-8")

    update_kwargs: dict = {}
    if updates:
        try:
            update_kwargs["updates"] = json.loads(updates)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON in --updates: {e}")
    elif key is not None:
        update_kwargs["key"] = key
        if value is not None:
            update_kwargs["value"] = value
    else:
        raise click.ClickException("Provide --key/--value or --updates")

    updated = apply_frontmatter_update(content, **update_kwargs)
    fp.write_text(updated, encoding="utf-8")
    click.echo(f"Updated frontmatter in: {file_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 12: replace_section
# ─────────────────────────────────────────────────────────────────────────────

@click.command("replace_section")
@click.argument("file_path")
@click.argument("heading")
@click.argument("content")
@click.option("--vault-path", help="Vault path override")
def replace_section_cmd(
    file_path: str,
    heading: str,
    content: str,
    vault_path: Optional[str],
):
    """Replace the body under a heading (preserves heading line)."""
    vp = vault_path_from_args(vault_path)
    fp = get_safe_file_path(str(vp), file_path)
    raw = fp.read_text(encoding="utf-8")
    rng = find_section_range(raw, heading)
    if not rng:
        raise click.ClickException(f"Heading not found: {heading}")
    updated = replace_section(raw, rng, content)
    fp.write_text(updated, encoding="utf-8")
    click.echo(f"Replaced section '{heading}' in: {file_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 13: replace_in_note
# ─────────────────────────────────────────────────────────────────────────────

@click.command("replace_in_note")
@click.argument("file_path")
@click.argument("old_text")
@click.option("--new-text", default="", help="Replacement text (default: empty)")
@click.option("--vault-path", help="Vault path override")
def replace_in_note_cmd(
    file_path: str,
    old_text: str,
    new_text: str,
    vault_path: Optional[str],
):
    """Replace the first occurrence of a text string in a note."""
    vp = vault_path_from_args(vault_path)
    fp = get_safe_file_path(str(vp), file_path)
    raw = fp.read_text(encoding="utf-8")
    updated = replace_in_note(raw, old_text, new_text)
    fp.write_text(updated, encoding="utf-8")
    click.echo(f"Replaced text in: {file_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 14: get_daily_note
# ─────────────────────────────────────────────────────────────────────────────

@click.command("get_daily_note")
@click.option("--vault-path", help="Vault path override")
def get_daily_note(vault_path: Optional[str]):
    """Get or create today's daily note."""
    vp = vault_path_from_args(vault_path)

    # Detect daily notes folder
    daily_folder = "Daily Notes" if (vp / "Daily Notes").is_dir() else ""

    date_str = datetime.now().strftime("%Y-%m-%d")
    file_name = f"{date_str}.md"
    fp = vp / daily_folder / file_name if daily_folder else vp / file_name

    if fp.exists():
        click.echo(fp.read_text(encoding="utf-8"))
    else:
        fp.parent.mkdir(parents=True, exist_ok=True)
        body = f"# {date_str}\n\n"
        fp.write_text(body, encoding="utf-8")
        click.echo(body)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 15: append_daily_log
# ─────────────────────────────────────────────────────────────────────────────

@click.command("append_daily_log")
@click.argument("heading")
@click.argument("content")
@click.option("--vault-path", help="Vault path override")
def append_daily_log(
    heading: str,
    content: str,
    vault_path: Optional[str],
):
    """Append text to a heading in today's daily note."""
    vp = vault_path_from_args(vault_path)
    daily_folder = "Daily Notes" if (vp / "Daily Notes").is_dir() else ""

    date_str = datetime.now().strftime("%Y-%m-%d")
    file_name = f"{date_str}.md"
    fp = vp / daily_folder / file_name if daily_folder else vp / file_name

    if not fp.exists():
        fp.parent.mkdir(parents=True, exist_ok=True)
        raw = f"# {date_str}\n\n"
        fp.write_text(raw, encoding="utf-8")
    else:
        raw = fp.read_text(encoding="utf-8")

    timestamp = datetime.now().strftime("%H:%M")
    entry = f"\n- [{timestamp}] {content}"

    rng = find_section_range(raw, heading)
    if rng:
        raw = raw[: rng.heading_end] + entry + raw[rng.heading_end :]
    else:
        raw += f"\n\n## {heading}{entry}"

    fp.write_text(raw, encoding="utf-8")
    click.echo(f"Appended to daily note under '{heading}'")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 16: rag_index
# ─────────────────────────────────────────────────────────────────────────────

@click.command("rag_index")
@click.option("--vault-path", help="Vault path override")
@click.option("--file-path", help="Index a specific file only")
@click.option("--force/--no-force", default=False, help="Force full reindex")
def rag_index(
    vault_path: Optional[str],
    file_path: Optional[str],
    force: bool,
):
    """Index the vault (or a file) for semantic search (RAG)."""
    vp = vault_path_from_args(vault_path)

    if file_path:
        result = _indexer.index_file(str(vp), file_path)
    else:
        result = _indexer.index_vault(str(vp), force=force)

    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Tool 17: rag_query
# ─────────────────────────────────────────────────────────────────────────────

@click.command("rag_query")
@click.argument("query")
@click.option("--vault-path", "vault_path", help="Vault path override")
@click.option("--limit", default=5, help="Max results (default: 5)")
def rag_query(query: str, vault_path: Optional[str], limit: int):
    """Semantic search on indexed vault."""
    if vault_path:
        config.set_vault(vault_path)
    results = _indexer.search(query, limit)
    if not results:
        click.echo("No results found.")
        return
    for r in results:
        click.echo("---")
        click.echo(f"File: {r.path}")
        click.echo(f"Score: {r.score:.4f}")
        click.echo(f"Content: {r.text}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

@click.group(
    invoke_without_command=True,
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.version_option("1.0.0")
@click.pass_context
def cli(ctx: click.Context):
    """obsidian-rag — Obsidian vault management and semantic search (RAG)."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register commands (excluding internal helpers)
for cmd in [
    set_vault,
    list_notes,
    read_note,
    create_note,
    append_note,
    search_notes,
    get_backlinks,
    get_links,
    get_broken_links,
    move_note,
    update_frontmatter,
    replace_section_cmd,
    replace_in_note_cmd,
    get_daily_note,
    append_daily_log,
    rag_index,
    rag_query,
]:
    cli.add_command(cmd)


def main():
    cli()


if __name__ == "__main__":
    main()
