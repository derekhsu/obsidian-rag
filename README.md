# obsidian-rag — Obsidian RAG Skill for OpenClaw

Chinese-optimized Obsidian vault manager with LanceDB + FastEmbed semantic search.

## Installation

```bash
cd ~/.openclaw/workspace/skills/obsidian-rag
pip3 install -r requirements.txt
```

Or install in development mode:

```bash
pip3 install -e .
```

## Quick Start

```bash
# 1. Set your vault path (saved to ~/.config/obsidian-rag/config.json)
python3 -m obsidian_rag.main set_vault /path/to/your/vault

# 2. Index the vault for semantic search
python3 -m obsidian_rag.main rag_index

# 3. Search semantically
python3 -m obsidian_rag.main rag_query "what did I discuss about the API?"

# 4. Browse notes
python3 -m obsidian_rag.main list_notes
python3 -m obsidian_rag.main read_note "Projects/Demo.md"

# 5. Write notes
python3 -m obsidian_rag.main create_note "Ideas/NewIdea.md" --content "# New Idea\n\n..."
python3 -m obsidian_rag.main append_note "Ideas/NewIdea.md" --content "\n- Another point"

# 6. Daily notes
python3 -m obsidian_rag.main append_daily_log "Work Log" "Finished the integration"
```

## Incremental Reindex

The `scripts/reindex.py` script provides advanced reindexing:

```bash
# Normal incremental reindex
python3 scripts/reindex.py /path/to/vault

# Full rebuild (ignores cached hashes)
python3 scripts/reindex.py /path/to/vault --force

# Watch mode (re-index changed files automatically)
python3 scripts/reindex.py /path/to/vault --watch
pip3 install watchdog  # required for --watch
```

## All Tools

| Command | Description |
|---------|-------------|
| `set_vault PATH` | Set default vault path |
| `list_notes [--subfolder FOLDER] [--limit N]` | List notes |
| `read_note FILE_PATH` | Read note content |
| `create_note FILE_PATH --content TEXT` | Create note |
| `append_note FILE_PATH --content TEXT` | Append to note |
| `search_notes QUERY` | Full-text search |
| `get_backlinks FILE_NAME` | Find backlinks |
| `get_links FILE_PATH` | Extract wikilinks |
| `get_broken_links [--subfolder FOLDER]` | Find broken links |
| `move_note SOURCE DEST` | Move/rename note |
| `update_frontmatter FILE_PATH --key K --value V` | Update frontmatter |
| `update_frontmatter FILE_PATH --updates '{"key":"value"}'` | Batch update |
| `replace_section FILE_PATH HEADING CONTENT` | Replace section body |
| `replace_in_note FILE_PATH OLD_TEXT [--new-text NEW]` | Replace text |
| `get_daily_note` | Get today's daily note |
| `append_daily_log HEADING CONTENT` | Log to daily note |
| `rag_index [--file PATH] [--force]` | Index for RAG |
| `rag_query QUERY [--limit N]` | Semantic search |

## Architecture

```
obsidian-rag/
├── SKILL.md            # This file — skill metadata
├── README.md           # Installation & usage guide
├── requirements.txt    # Python dependencies
├── config.py           # Configuration & vault path
├── main.py             # CLI entry point (16 tools)
├── utils.py            # Wikilink, frontmatter, section utils
├── chunking.py         # Chinese-optimized text chunking
├── vector_store.py     # LanceDB + FastEmbed RAG engine
└── scripts/
    └── reindex.py      # Incremental reindex script
```

## Configuration

| File | Location | Purpose |
|------|----------|---------|
| Config | `~/.config/obsidian-rag/config.json` | Vault path |
| Vector DB | `~/.local/share/obsidian-rag/lancedb/` | LanceDB data |
| Hash cache | `~/.local/share/obsidian-rag/file_hashes.json` | Incremental index |

## Tech Stack

- **Chunking:** Chinese punctuation (`。！？；：`) + ASCII punctuation
- **Embedding:** FastEmbed + `BAAI/bge-small-zh-v1.5` (512-dim)
- **Vector store:** LanceDB (local, embedded)
- **Python:** 3.9+
