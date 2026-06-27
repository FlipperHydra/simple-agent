# simple-agent

A lightweight, extensible AI agent built on [Ollama](https://ollama.com). The agent streams responses, detects tool calls in real time, executes them, and feeds results back into the conversation -- all in a persistent REPL session.

---

## Features

- Streaming responses with live thinking output
- XML-based tool call parsing -- no JSON schemas or function-calling APIs required
- Dynamic tool registration -- add a new tool without touching the processor
- Persistent conversation history with `/clear` and `/quit` REPL commands
- Tool result injection -- the model reasons about what tools return before producing a final answer
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

---

## Requirements

```
ollama
duckduckgo-search
```

---

## Available Tools

The agent ships with six built-in tools. All tools are registered automatically -- the model is informed of every tool and its arguments at startup.

| Tool | Arguments | Description |
|---|---|---|
| `write_tool` | `text` | Appends text to `output.txt` in the current directory |
| `save_tool` | `filename`, `content` | Appends content to any named file |
| `read_file` | `filename` | Reads and returns the full contents of a file |
| `list_files` | `directory` | Lists all entries in a directory (defaults to `.`) |
| `fetch_url` | `url` | Fetches and returns the first 4000 characters of a URL |
| `search_web` | `query` | Searches the web via DuckDuckGo and returns top 5 results |

---

## Tool Call Format

The model emits tool calls as XML blocks inside its response. Each argument is wrapped in a numbered tag:

```
<tool_name>
<arg1>first argument</arg1>
<arg2>second argument</arg2>
</tool_name>
```

The tool processor detects complete blocks as they arrive in the stream, dispatches the call, and injects the result back into the conversation as a follow-up user message so the model can reason about it.

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
| `/clear` | Resets conversation history, keeps system prompts |
| `/quit` | Exits the agent and prints session complete message |

---

## Project Structure

```
simple-agent/
├── main.py            # Entry point, REPL loop, async chat orchestration
├── prompts.py         # System prompts -- tool list, format rules, reasoning guide
├── tool_registry.py   # Tool registration and factory functions
├── tool_processor.py  # Streaming XML parser and async tool dispatcher
├── tools.py           # Tool implementations as static methods
└── requirements.txt
```

---

## Model

The default model is `gemma4`. To change it, update the `model` parameter in `main.py`:

```python
response = await _client.chat(
    model='your-model-here',
    ...
)
```

Any model available in your local Ollama instance will work. Models that support a `think` field will display intermediate reasoning inline.

---

## License

MIT
