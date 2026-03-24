#!/usr/bin/env python3
"""
Incremental reindex script for obsidian-rag.

Usage:
    # Full reindex
    python scripts/reindex.py /path/to/vault

    # Force full rebuild
    python scripts/reindex.py /path/to/vault --force

    # Index a specific file
    python scripts/reindex.py /path/to/vault --file "Daily Notes/2024-01-01.md"

    # Watch mode (re-index on file change) — requires watchdog
    python scripts/reindex.py /path/to/vault --watch
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure the skill package is importable (pointing to ../src)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from obsidian_rag.vector_store import VaultIndexer
from obsidian_rag.config import config


def main():
    parser = argparse.ArgumentParser(description="obsidian-rag incremental reindex")
    parser.add_argument("vault_path", help="Path to Obsidian vault")
    parser.add_argument("--force", action="store_true", help="Force full rebuild")
    parser.add_argument("--file", dest="file_path", help="Index a specific file only")
    parser.add_argument("--watch", action="store_true", help="Watch mode (requires watchdog)")
    args = parser.parse_args()

    vault = Path(args.vault_path).resolve()
    if not vault.is_dir():
        print(f"[reindex] Error: not a directory: {vault}", file=sys.stderr)
        sys.exit(1)

    # Set vault path in config so it persists
    config.set_vault(str(vault))

    indexer = VaultIndexer()

    if args.watch:
        _watch_mode(indexer, str(vault))
    elif args.file_path:
        result = indexer.index_file(str(vault), args.file_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        result = indexer.index_vault(str(vault), force=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2))


def _watch_mode(indexer: VaultIndexer, vault: str):
    """Watch vault for file changes and re-index incrementally."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    except ImportError:
        print(
            "[reindex] Error: --watch requires 'watchdog' package.\n"
            "  pip install watchdog",
            file=sys.stderr,
        )
        sys.exit(1)

    class ChangeHandler(FileSystemEventHandler):
        def __init__(self, indexer_: VaultIndexer, vault_: str):
            self.indexer = indexer_
            self.vault = vault_
            self._debounce: dict[str, float] = {}
            self._debounce_seconds = 2.0

        def on_modified(self, event):
            if event.is_directory:
                return
            if not event.src_path.endswith(".md"):
                return

            # Debounce
            import time
            now = time.time()
            key = event.src_path
            if key in self._debounce and now - self._debounce[key] < self._debounce_seconds:
                return
            self._debounce[key] = now

            rel = str(Path(event.src_path).relative_to(self.vault))
            print(f"[reindex] File changed: {rel}")
            try:
                result = self.indexer.index_file(self.vault, rel)
                print(f"[reindex] Re-indexed: {rel} → {result.get('chunks', 0)} chunks")
            except Exception as e:
                print(f"[reindex] Error re-indexing {rel}: {e}", file=sys.stderr)

    handler = ChangeHandler(indexer, vault)
    observer = Observer()
    observer.schedule(handler, vault, recursive=True)
    observer.start()
    print(f"[reindex] Watching {vault} for .md changes... (Ctrl+C to stop)")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
