"""Configuration for obsidian-rag skill."""
from __future__ import annotations

import json
import os
import platformdirs
from pathlib import Path
from typing import Optional

# ── Paths ────────────────────────────────────────────────────────────────────

APP_NAME = "obsidian-rag"
APP_AUTHOR = "openclaw"

def _get_config_dir() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME, APP_AUTHOR))

def _get_data_dir() -> Path:
    return Path(platformdirs.user_data_dir(APP_NAME, APP_AUTHOR))

CONFIG_DIR = _get_config_dir()
DATA_DIR = _get_data_dir()
CONFIG_FILE = CONFIG_DIR / "config.json"
DB_PATH = DATA_DIR / "lancedb"
HASH_PATH = DATA_DIR / "file_hashes.json"

# ── RAG Defaults ─────────────────────────────────────────────────────────────

# FastEmbed / BAAI model
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512

# Chunking (characters)
MIN_CHUNK_CHARS = 40
MAX_CHUNK_CHARS = 1800
TARGET_CHUNK_CHARS = 700

# Indexing
EMBED_BATCH_SIZE = 8          # Conservative for memory-limited environments
FILE_READ_CONCURRENCY = 50
DELETE_BATCH = 100

# ── Config helpers ────────────────────────────────────────────────────────────

class Config:
    """Runtime configuration — loaded from config.json and environment."""

    vault_path: Optional[str] = None

    def __init__(self):
        self.vault_path = os.environ.get("OBSIDIAN_VAULT_PATH")
        self._load_from_file()

    def _load_from_file(self):
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                if not self.vault_path and data.get("vault_path"):
                    self.vault_path = data["vault_path"]
            except Exception:
                pass

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps({"vault_path": self.vault_path}, ensure_ascii=False),
            encoding="utf-8",
        )

    def set_vault(self, path: str):
        self.vault_path = str(Path(path).resolve())
        self.save()


config = Config()
