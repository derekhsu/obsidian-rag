"""LanceDB vector store with FastEmbed embeddings for Obsidian RAG."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

from obsidian_rag.chunking import Chunk, ChunkOptions, build_chunks
from obsidian_rag.config import (
    CONFIG_DIR,
    DATA_DIR,
    DB_PATH,
    EMBED_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FILE_READ_CONCURRENCY,
    HASH_PATH,
    DELETE_BATCH,
    MIN_CHUNK_CHARS,
    MAX_CHUNK_CHARS,
    TARGET_CHUNK_CHARS,
)

# ── LanceDB Schema ─────────────────────────────────────────────────────────────

class NoteChunk(LanceModel):
    """LanceDB row schema for a note chunk."""
    id: str
    path: str
    chunk_index: int
    text: str
    vector: Vector(EMBEDDING_DIM)


# ── VaultIndexer ───────────────────────────────────────────────────────────────

@dataclass
class IndexResult:
    success: bool
    chunks: int = 0
    message: str = ""


@dataclass
class SearchResult:
    path: str
    text: str
    score: float  # similarity score


class VaultIndexer:
    """Index and search an Obsidian vault using LanceDB + FastEmbed."""

    def __init__(self):
        self._db: Optional[lancedb.DB] = None
        self._table: Optional[lancedb.Table] = None

    # ── LanceDB setup ─────────────────────────────────────────────────────────

    def _get_db(self) -> lancedb.DB:
        if self._db is None:
            self._db = lancedb.connect(str(DB_PATH))
        return self._db

    def _get_table(self) -> lancedb.Table:
        if self._table is None:
            db = self._get_db()
            if "notes" in db.table_names():
                self._table = db.open_table("notes")
            else:
                self._table = self._create_table(db)
        return self._table

    def _create_table(self, db: lancedb.DB) -> lancedb.Table:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Create empty table with schema
        try:
            return db.create_table("notes", schema=NoteChunk)
        except Exception:
            # Table may already exist
            return db.open_table("notes")

    # ── Chunking options ───────────────────────────────────────────────────────

    @staticmethod
    def _chunking_options() -> ChunkOptions:
        return ChunkOptions(
            min_chars=MIN_CHUNK_CHARS,
            max_chars=MAX_CHUNK_CHARS,
            target_chars=TARGET_CHUNK_CHARS,
        )

    # ── Index single file ─────────────────────────────────────────────────────

    async def index_file_async(
        self,
        vault_path: str,
        relative_path: str,
    ) -> IndexResult:
        """Index or re-index a single file.

        Replaces all existing chunks for that path.
        """
        import frontmatter

        file_path = Path(vault_path) / relative_path
        if not file_path.exists():
            return IndexResult(success=False, message=f"File not found: {relative_path}")

        try:
            raw = file_path.read_text(encoding="utf-8")
            post = frontmatter.parse(raw)
            body = post[1] if post else raw
        except UnicodeDecodeError as e:
            return IndexResult(success=False, message=f"Encoding error: {e}")

        chunks = build_chunks(relative_path, body, self._chunking_options())
        if not chunks:
            return IndexResult(success=True, chunks=0, message="No content to index.")

        # Embed in batches
        vectors = await self._embed_batch([c.text for c in chunks])

        rows = [
            {
                "id": c.id,
                "path": c.path,
                "chunk_index": c.index,
                "text": c.text,
                "vector": vectors[i],
            }
            for i, c in enumerate(chunks)
        ]

        db = self._get_db()
        if "notes" not in db.table_names():
            self._table = db.create_table("notes", data=rows)
        else:
            self._table = db.open_table("notes")
            # Delete old chunks for this path
            try:
                _safe = relative_path.replace("\\", "\\\\").replace("'", "''")
                self._table.delete(f"path = '{_safe}'")
            except Exception:
                pass
            self._table.add(rows)

        return IndexResult(success=True, chunks=len(chunks))

    def index_file(self, vault_path: str, relative_path: str) -> dict:
        """Synchronous wrapper for index_file_async."""
        return asyncio.run(self.index_file_async(vault_path, relative_path)).__dict__

    # ── Index entire vault ────────────────────────────────────────────────────

    async def index_vault_async(
        self,
        vault_path: str,
        force: bool = False,
        verbose: bool = True,
    ) -> IndexResult:
        """Index all markdown files in the vault.

        Incremental by default: skips files whose MD5 hash hasn't changed.
        Use force=True to rebuild the entire index from scratch.
        """
        import frontmatter
        from tqdm.asyncio import tqdm as atqdm

        vault = Path(vault_path)
        # Collect all .md files
        md_files = sorted(vault.rglob("*.md"))
        total_files = len(md_files)
        if verbose:
            print(f"[obsidian-rag] Found {total_files} notes in {vault_path}", file=sys.stderr)

        # Load previous hashes
        prev_hashes: dict[str, str] = {}
        if not force and HASH_PATH.exists():
            try:
                prev_hashes = json.loads(HASH_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass

        can_incremental = bool(prev_hashes) and not force

        # Phase 1: Read files, compute hashes, build chunks
        all_texts: list[str] = []
        all_meta: list[dict] = []   # {relative_path, chunk_index}
        current_hashes: dict[str, str] = {}
        changed_paths: list[str] = []
        skipped = 0

        def _compute_hash(content: bytes) -> str:
            import hashlib
            return hashlib.md5(content).hexdigest()

        async def _read_file(fp: Path) -> Optional[tuple[str, str, list[Chunk]]]:
            """Returns (relative_path, body, chunks) or None on skip."""
            rel = str(fp.relative_to(vault))
            content_bytes = fp.read_bytes()
            content_hash = _compute_hash(content_bytes)
            current_hashes[rel] = content_hash

            if can_incremental and prev_hashes.get(rel) == content_hash:
                return None  # Skip unchanged

            try:
                raw = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return None

            post = frontmatter.parse(raw)
            body = post[1] if post else raw

            chunks = build_chunks(rel, body, self._chunking_options())
            return (rel, body, chunks)

        semaphore = asyncio.Semaphore(FILE_READ_CONCURRENCY)

        async def _read_with_limit(fp: Path) -> Optional[tuple]:
            async with semaphore:
                return await _read_file(fp)

        if verbose:
            tasks = [_read_with_limit(fp) for fp in md_files]
            results = await atqdm.gather(*tasks, desc="Reading files")
        else:
            results = await asyncio.gather(*[_read_with_limit(fp) for fp in md_files])

        for result in results:
            if result is None:
                skipped += 1
                continue
            rel, body, chunks = result
            changed_paths.append(rel)
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk.text)
                all_meta.append({"path": rel, "index": i})

        deleted_paths = list(set(prev_hashes.keys()) - set(current_hashes.keys()))

        if verbose:
            print(
                f"[obsidian-rag] Incremental={can_incremental}: "
                f"{len(changed_paths)} changed, {len(deleted_paths)} deleted, {skipped} unchanged",
                file=sys.stderr,
            )

        # Early exit: nothing changed
        if can_incremental and not all_texts and not deleted_paths:
            HASH_PATH.write_text(json.dumps(current_hashes), encoding="utf-8")
            return IndexResult(success=True, chunks=0, message="Index up to date.")

        if not all_texts and not deleted_paths:
            return IndexResult(success=False, message="No content found to index.")

        # Phase 2: Embed and write
        db = self._get_db()
        table_exists = "notes" in db.table_names()

        # Delete old chunks for changed/deleted files
        if table_exists:
            self._table = db.open_table("notes")
            paths_to_delete = list(set(changed_paths) | set(deleted_paths))
            def _esc(p: str) -> str:
                return p.replace("\\", "\\\\").replace("'", "''")

            for i in range(0, len(paths_to_delete), DELETE_BATCH):
                batch = paths_to_delete[i : i + DELETE_BATCH]
                escaped = [f"'{_esc(p)}'" for p in batch]
                try:
                    self._table.delete(f"path IN ({', '.join(escaped)})")
                except Exception:
                    pass

        # Sort by text length to reduce padding waste
        sorted_pairs = sorted(zip(all_texts, all_meta), key=lambda x: len(x[0]))
        sorted_texts, sorted_meta = zip(*sorted_pairs) if sorted_pairs else ([], [])

        indexed = 0
        accumulated_rows: list[dict] = []

        async def _embed_and_persist():
            nonlocal indexed, accumulated_rows

            for i in range(0, len(sorted_texts), EMBED_BATCH_SIZE):
                batch_texts = list(sorted_texts[i : i + EMBED_BATCH_SIZE])
                batch_meta = list(sorted_meta[i : i + EMBED_BATCH_SIZE])

                # Embed
                vectors = await self._embed_batch(batch_texts)

                for j, vec in enumerate(vectors):
                    m = batch_meta[j]
                    accumulated_rows.append({
                        "id": _md5_chunk_id(m["path"], m["index"]),
                        "path": m["path"],
                        "chunk_index": m["index"],
                        "text": batch_texts[j],
                        "vector": vec,
                    })

                indexed += len(batch_texts)
                if verbose:
                    pct = min(100, int(indexed / len(sorted_texts) * 100))
                    bar = "#" * (pct // 3) + "-" * (33 - pct // 3)
                    print(
                        f"\r[obsidian-rag] Embedding [{bar}] {pct}% ({indexed}/{len(sorted_texts)})",
                        file=sys.stderr,
                        end="",
                    )

                # Persist every ~5 batches
                if len(accumulated_rows) >= EMBED_BATCH_SIZE * 5:
                    await asyncio.to_thread(self._persist_rows, accumulated_rows)
                    accumulated_rows = []

            if accumulated_rows:
                await asyncio.to_thread(self._persist_rows, accumulated_rows)

            if verbose:
                print(file=sys.stderr)

        await _embed_and_persist()

        # Save hashes
        HASH_PATH.parent.mkdir(parents=True, exist_ok=True)
        HASH_PATH.write_text(json.dumps(current_hashes), encoding="utf-8")

        msg = (
            f"Indexed {indexed} chunks from {len(changed_paths)} files, "
            f"removed {len(deleted_paths)} files."
        )
        if verbose:
            print(f"[obsidian-rag] {msg}", file=sys.stderr)
        return IndexResult(success=True, chunks=indexed, message=msg)

    def _persist_rows(self, rows: list[dict]):
        """Write rows to LanceDB table (called in thread)."""
        if not rows:
            return
        db = self._get_db()
        if "notes" not in db.table_names():
            self._table = db.create_table("notes", data=rows)
        else:
            self._table = db.open_table("notes")
            self._table.add(rows)

    def index_vault(self, vault_path: str, force: bool = False) -> dict:
        """Synchronous wrapper for index_vault_async."""
        return asyncio.run(self.index_vault_async(vault_path, force)).__dict__

    # ── Embedding ─────────────────────────────────────────────────────────────

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using FastEmbed."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _sync_embed, texts
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        """Semantic search across indexed notes.

        Returns top-N results with path, text, and similarity score.
        """
        try:
            table = self._get_table()
        except Exception:
            return []

        # Embed query
        vector = _sync_embed([query])[0]

        try:
            results = (
                table.search(vector)
                .limit(limit)
                .to_list()
            )
        except Exception:
            return []

        output = []
        for r in results:
            # LanceDB returns distance; convert to similarity (higher=better)
            dist = r.get("distance", 0.0)
            score = 1.0 / (1.0 + dist) if dist is not None else 0.0
            output.append(
                SearchResult(
                    path=r.get("path", ""),
                    text=r.get("text", ""),
                    score=score,
                )
            )
        return output


# ── Module-level embedding singleton ───────────────────────────────────────────

_embedder = None
_model_name = EMBEDDING_MODEL

def _get_embedder():
    global _embedder
    if _embedder is None:
        import fastembed
        _embedder = fastembed.TextEmbedding(model_name=_model_name)
    return _embedder


def _sync_embed(texts: list[str]) -> list[list[float]]:
    """Thread-safe synchronous embedding using FastEmbed."""
    emb = _get_embedder()
    return [vec.tolist() for vec in emb.embed(texts)]


def _md5_chunk_id(path: str, index: int) -> str:
    import hashlib
    return hashlib.md5(f"{path}-{index}".encode("utf-8")).hexdigest()
