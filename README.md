# simple-agent

A lightweight, extensible AI agent built on [Ollama](https://ollama.com). The agent streams responses, detects tool calls in real time, executes them, and feeds results back into the conversation -- all in a persistent REPL session. Conversation history is persisted to disk automatically and kept within the model's context window through token-budgeted compaction.

---

## Features

- Streaming responses with live thinking output
- XML-based tool call parsing -- no JSON schemas or function-calling APIs required
- Dynamic tool registration -- add a new tool without touching the processor
- 23 built-in tools (files, JSON, memory, web, math, iterative summarization, soul editing)
- **Persistent conversation** -- history autoloads at startup and autosaves after every turn
- **Context compaction** -- older messages are summarized into a compact system note instead of being silently dropped
- Crash-safe atomic writes for all persistent state
- Async runtime throughout

---

## Setup

```bash
git clone https://github.com/FlipperHydra/simple-agent
cd simple-agent
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Ollama must be running locally with your chosen model pulled:

```bash
ollama pull gemma4
ollama serve
```

Then run the agent:

```bash
python main.py
```

At startup the agent validates that the configured model is available locally and fails fast with an `ollama pull <model>` hint if it is not.

---

## Requirements

```
ollama
ddgs
tiktoken
```

- `ollama` -- Ollama Python client (chat/streaming, model listing)
- `ddgs` -- DuckDuckGo search backend used by `search_web`
- `tiktoken` -- token counting for context compaction (compaction gracefully degrades to a rough estimate if unavailable)

Everything else the agent uses is from the Python standard library.

---

## Configuration (environment variables)

All settings have sensible defaults; override them via environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `AGENT_MODEL` | `gemma4` | Ollama model to use. `gemma4` is a real Ollama model and is the default. |
| `AGENT_DATA_DIR` | `./data` (local), `/app/data` (container) | Directory holding all persistent state (`memory.json`, `soul.md`, `conversation.json`, session snapshots, `output.txt`). |
| `AGENT_NUM_CTX` | `16384` | Context window (tokens) passed to Ollama, and the hard cap that triggers automatic compaction. |
| `AGENT_COMPACT_THRESHOLD` | `2000` | Token count above which the agent offers to compact older messages. |
| `AGENT_COMPACT_REPROMPT_DELTA` | `250` | After you accept or decline a compaction prompt, the agent will not ask again until the conversation grows by more than this many tokens. |
| `OLLAMA_HOST` | (Ollama default) | Read by the Ollama client to locate the server (e.g. `http://host.docker.internal:11434`). |

---

## Conversation Persistence

The conversation is stored as JSON at `<AGENT_DATA_DIR>/conversation.json`.

- **Autoload:** on startup the agent restores the previous conversation from disk.
- **Autosave:** after every turn (and after compaction) the conversation is written back atomically.

JSON was chosen over a database because the conversation is a small, append-mostly list of message dicts -- direct serialization, human-inspectable, no schema/migration cost.

You can also take named snapshots with `/save_session` (writes `session_<timestamp>.json` into the data dir) and restore one with `/load_session <file>`. These are manual snapshots layered on top of the automatic persistence.

---

## Context Compaction

Instead of hard-truncating by message count, the agent keeps the conversation within budget by **summarizing** older turns:

- After each turn the total conversation token count is measured with `tiktoken`.
- When it exceeds `AGENT_COMPACT_THRESHOLD` (default 2000), the agent prompts:

  ```
  Conversation context exceeding 2000 tokens. In order to preserve current context, compact older messages? Y/N
  ```

  - **Y** -- roughly the oldest half of the conversation (split on a user-message boundary so a user/assistant pair is never torn apart) is summarized by the model into a single `system` message tagged `[Compacted summary of earlier conversation]`. The most recent turns are kept verbatim. If the result is still over threshold, it compacts again.
  - **N** -- the conversation is left as-is. The agent will **not** re-prompt every turn; it waits until the conversation grows by more than `AGENT_COMPACT_REPROMPT_DELTA` tokens (default 250) since your last decision.
- **Hard cap:** if the conversation would exceed `AGENT_NUM_CTX`, compaction runs **automatically** (no prompt) to prevent context corruption, printing an `[auto-compact]` notice.

Summaries use a dense factual prompt that preserves names, decisions, code/state references, and open threads.

---

## Available Tools

The agent ships with **23 built-in tools**, all registered automatically -- the model is informed of every tool and its arguments at startup. Tools marked **[DANGEROUS]** require interactive confirmation before they run.

| Tool | Arguments | Description |
|---|---|---|
| `write_tool` | `text` | Appends text to `output.txt` in the data directory |
| `save_tool` | `filename`, `content` | Appends content to a named file |
| `overwrite_file` | `filename`, `content` | Replaces the entire contents of a file |
| `get_datetime` | -- | Returns the current date and time |
| `read_file` | `filename` | Reads and returns the full contents of a file |
| `list_files` | `directory` | Lists entries in a directory (defaults to `.`) |
| `fetch_url` | `url` | Fetches and returns the first 4000 characters of a URL (http/https only; see security note) |
| `search_web` | `query` | Searches the web via DDGS and returns the top 5 results |
| `delete_file` | `filename` | Permanently deletes a file **[DANGEROUS]** |
| `make_directory` | `path` | Creates a directory and any missing parents |
| `copy_file` | `src`, `dest` | Copies a file, leaving the original intact |
| `move_file` | `src`, `dest` | Moves or renames a file **[DANGEROUS]** |
| `append_memory` | `note` | Appends a timestamped note to `memory.json` |
| `recall_memory` | -- | Returns all facts and log entries from `memory.json` |
| `clear_memory` | -- | Wipes all entries from `memory.json` **[DANGEROUS]** |
| `write_json` | `filename`, `key`, `value` | Upserts a key/value pair into a JSON file |
| `read_json` | `filename`, `key` | Reads a key from a JSON file (or the whole file if key omitted) |
| `eval_math` | `expression` | Safely evaluates a math expression via an AST whitelist (see safety note) |
| `summarize_file` | `filename` | Init step: reads a file, computes chunks, returns dispatch instructions |
| `summarize_file_chunk` | `filename`, `chunk_index`, `chunk_summary` | Stores a per-chunk summary and returns the next chunk |
| `summarize_file_finalize` | `filename` | Synthesizes all chunk summaries into a final file summary |
| `zip_files` | `filenames_csv`, `output_zip` | Compresses a comma-separated list of files into a zip archive |
| `propose_soul_edit` | `section`, `proposed_content` | Proposes an update to a `soul.md` section for user approval |

### Security notes

- **`fetch_url`** only accepts `http` and `https` URLs (schemes like `file://` and `ftp://` are rejected) and, after DNS resolution, refuses to connect to private, loopback, or link-local addresses (e.g. `127.0.0.1`, `10.0.0.0/8`, `192.168.0.0/16`, `169.254.0.0/16`, `::1`). This is an SSRF / local-file-read guard.
- **`eval_math`** uses an AST whitelist (no names, calls, or attribute access) and additionally bounds exponentiation: a power with an exponent above `1000`, or a computed/nested exponent such as `9 ** 9 ** 9`, is refused with a clear message rather than allowed to allocate unbounded memory.

---

## Memory & Soul

- **`memory.json`** -- persistent key/value facts plus a timestamped note log, managed by the `append_memory` / `recall_memory` / `clear_memory` / `write_json` / `read_json` tools.
- **`soul.md`** -- the agent's persona / user-profile document. It is seeded from the shipped copy on first run and edited through `/soul_update` and the soul-edit tools.

### `/soul_update` output format

`/soul_update` asks the model to review memory and rewrite the User Profile in `soul.md`. The model **must** wrap the updated document between explicit sentinels:

```
===SOUL START===
# ... full soul.md content ...
===SOUL END===
```

Only the text between the two sentinels is written. If either sentinel is missing (or the extracted content is not a valid soul document), the update is aborted and `soul.md` is left untouched. Writes are atomic and the previous version is preserved as `soul.md.bak`. This prevents the earlier failure mode where any stray `# ` heading in the model's commentary could corrupt the file.

---

## Tool Call Format

The model emits tool calls as XML blocks inside its response. Each argument is wrapped in a numbered tag:

```
<tool_name>
<arg1>first argument</arg1>
<arg2>second argument</arg2>
</tool_name>
```

The tool processor detects complete blocks as they arrive in the stream, dispatches the call, and injects the result back into the conversation as a `tool`-role message so the model can reason about it.

---

## Adding a Tool

**Step 1 -- Implement it in `tools.py`:**

```python
@staticmethod
def my_tool(input: str) -> str:
    return f'processed: {input}'
```

**Step 2 -- Register it in `tool_registry.py`:**

```python
def _make_my_tool() -> Callable:
    async def my_tool(input: str) -> str:
        return Tools.my_tool(input)
    return my_tool
```

```python
self.register_tool(
    'my_tool',
    _make_my_tool(),
    arg_names=['input'],
    description='describe what this tool does',
)
```

That is all -- the tool prompt, XML parser, and result handler update automatically.

---

## REPL Commands

| Command | Effect |
|---|---|
| `/?` | Show REPL commands and registered tools |
| `/tools` | List all registered tools |
| `/soul` | Print `soul.md` |
| `/soul_reset` | Restore `soul.md` to default content (backs up the old one) |
| `/soul_update` | Review memory and update the User Profile in `soul.md` |
| `/save_session` | Save the current conversation to a timestamped JSON snapshot |
| `/load_session <f>` | Load a saved session from JSON |
| `/model <name>` | Switch the active Ollama model for this session |
| `/history` | Show a summary of the current conversation history |
| `/clear` | Clear conversation history (keeps system prompts) |
| `/quit` | Exit the agent |

---

## Docker

State is persisted through a **single data-directory volume** mounted at `/app/data`, driven by `AGENT_DATA_DIR`. (The previous setup mounted individual files as volumes; Docker named volumes are directories, so that silently turned `memory.json` / `soul.md` into directories and broke every read/write. A single directory volume avoids this entirely.)

```bash
docker compose up --build
```

`docker-compose.yml` sets `AGENT_DATA_DIR=/app/data`, points `OLLAMA_HOST` at your host's Ollama instance, and selects the model via `AGENT_MODEL`. On Linux you may need `--add-host` for `host.docker.internal` or can switch to host networking (see the commented options in the compose file).

---

## Project Structure

```
simple-agent/
├── main.py            # Entry point, REPL loop, persistence, compaction, chat orchestration
├── config.py          # Data-directory paths and crash-safe atomic write helpers
├── prompts.py         # System prompts -- tool list, format rules, reasoning + compaction prompts
├── tool_registry.py   # Tool registration and factory functions
├── tool_processor.py  # Streaming XML parser and async tool dispatcher
├── tools.py           # Tool implementations as static methods
├── tests/             # Import smoke, compaction, and security tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Model

The default model is `gemma4` and is configurable via the `AGENT_MODEL` environment variable (or `/model <name>` at runtime). Any model available in your local Ollama instance will work; models that support a `think` field will display intermediate reasoning inline. The model is validated at startup and the agent exits with a helpful message if it has not been pulled.

---

## License

MIT
