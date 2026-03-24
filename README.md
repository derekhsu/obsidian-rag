# obsidian-rag — Obsidian RAG Agent Skill

An AI agent skill for managing Obsidian vaults, featuring Chinese-optimized text chunking and LanceDB + FastEmbed semantic search.

**Acknowledgment:** This project is inspired by [gemini-obsidian](https://github.com/thoreinstein/gemini-obsidian).

## Installation

Install via the standard agent skills CLI (`npx skills`), regardless of which specific AI agent environment you are using:

```bash
npx skills add derekhsu/obsidian-rag@obsidian-rag
# Or via full repository URL:
# npx skills add https://github.com/derekhsu/obsidian-rag@obsidian-rag
```

Once installed by your AI agent, the skill handles setting up its own Python environment via `uv`.

## Architecture

This project is structured as a self-contained AI agent skill, conforming to the standard `agent-skills` specification. All execution logic, metadata, and dependencies are encapsulated in the `skills/obsidian-rag/` directory to ensure perfect portability.

```text
obsidian-rag/
└── skills/
    └── obsidian-rag/
        ├── SKILL.md            # Agent instructions and metadata
        ├── pyproject.toml      # Python dependencies (managed by uv)
        ├── uv.lock             # Dependency lockfile
        ├── scripts/
        │   └── reindex.py      # Background/Incremental reindexing script
        └── src/
            └── obsidian_rag/   # Core Python package
                ├── main.py     # CLI entry point (16 tools)
                ├── utils.py    # Wikilink, frontmatter, section utils
                ├── chunking.py # Chinese-optimized text chunking
                └── vector_store.py # LanceDB + FastEmbed RAG engine
```

## Agent Usage & Tools

Once the skill is activated, the agent manages execution by running `uv run` within the `skills/obsidian-rag` directory.

### Available Tools

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

## Configuration & Storage

Storage paths are OS-dependent (managed via `platformdirs`):

| File | Location (macOS example) | Purpose |
|------|----------|---------|
| Config | `~/Library/Application Support/obsidian-rag/config.json` | Vault path |
| Vector DB | `~/.local/share/obsidian-rag/lancedb/` | LanceDB data |
| Hash cache | `~/.local/share/obsidian-rag/file_hashes.json` | Incremental index caching |

## Manual Development

If you wish to develop or test this skill locally:

```bash
git clone https://github.com/derekhsu/obsidian-rag.git
cd obsidian-rag/skills/obsidian-rag
uv sync

# Run tools
uv run python -m obsidian_rag list_notes

# Reindex vault
python scripts/reindex.py /path/to/vault
```
