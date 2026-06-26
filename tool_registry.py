import os
import json
import urllib.request
import urllib.parse
from typing import Any, Callable, Dict, List, NamedTuple


# Tools

def save_tool(filename: str, content: str) -> None:
    with open(filename, "a", encoding="utf-8") as f:
        f.write(content + "\n")
    print(f'\n[save_tool] Saved \u2192 {filename}')


def write_tool(text: str) -> None:
    formatted = (
        text
        .replace('\\n', '\n')
        .replace('\\t', '\t')
        .replace('\\\'', '\'')
    )
    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(formatted + "\n")
    print(f'\n[write_tool] Wrote \u2192 {formatted}')


def read_file(filename: str) -> str:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"[read_file] File not found: {filename}"


def list_files(directory: str = ".") -> str:
    try:
        entries = os.listdir(directory)
        return "\n".join(entries)
    except FileNotFoundError:
        return f"[list_files] Directory not found: {directory}"


def fetch_url(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.read().decode("utf-8", errors="replace")[:4000]
    except Exception as e:
        return f"[fetch_url] Error: {e}"


def search_web(query: str) -> str:
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        abstract = data.get("AbstractText", "")
        results = [r.get("Text", "") for r in data.get("RelatedTopics", [])[:5]]
        combined = abstract + "\n" + "\n".join(results)
        return combined.strip() or "[search_web] No results found."
    except Exception as e:
        return f"[search_web] Error: {e}"


# Tool metadata


class ToolMeta(NamedTuple):
    func: Callable[..., Any]
    arg_names: List[str]   # ordered list of argument labels shown in the prompt
    description: str       # one-line description shown in the prompt


# Tool Registry


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolMeta] = {}
        self._register_defaults()

    # MODIFY THIS TO ADD TOOLS
    def _register_defaults(self) -> None:
        self.register_tool(
            "write_tool",
            write_tool,
            arg_names=["text"],
            description="writes the given text to output.txt",
        )
        self.register_tool(
            "save_tool",
            save_tool,
            arg_names=["filename", "content"],
            description="saves content to a named file",
        )
        self.register_tool(
            "read_file",
            read_file,
            arg_names=["filename"],
            description="reads and returns the contents of a file",
        )
        self.register_tool(
            "list_files",
            list_files,
            arg_names=["directory"],
            description="lists all files in a directory",
        )
        self.register_tool(
            "fetch_url",
            fetch_url,
            arg_names=["url"],
            description="fetches and returns the text content of a URL (max 4000 chars)",
        )
        self.register_tool(
            "search_web",
            search_web,
            arg_names=["query"],
            description="searches DuckDuckGo and returns a summary of results",
        )

    def register_tool(
        self,
        name: str,
        func: Callable[..., Any],
        *,
        arg_names: List[str] | None = None,
        description: str = "",
    ) -> None:
        self._tools[name] = ToolMeta(
            func=func,
            arg_names=arg_names or ["arg"],
            description=description,
        )

    def get(self, name: str) -> Callable[..., Any] | None:
        meta = self._tools.get(name)
        return meta.func if meta else None

    def meta(self, name: str) -> ToolMeta | None:
        return self._tools.get(name)

    def all(self) -> Dict[str, ToolMeta]:
        return dict(self._tools)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def tag_descriptions(self) -> List[str]:
        """
        Returns one description block per registered tool showing every
        argument in nested-tag format, e.g.:

          write_tool — writes the given text to a file
            <write_tool>
              <arg1>text</arg1>
            </write_tool>

        Used by build_tool_prompt() to auto-generate the system prompt.
        """
        blocks = []
        for name, meta in self._tools.items():
            inner = "\n".join(
                f"  <arg{i+1}>{label}</arg{i+1}>"
                for i, label in enumerate(meta.arg_names)
            )
            blocks.append(
                f"  {name} \u2014 {meta.description}\n"
                f"  <{name}>\n{inner}\n  </{name}>"
            )
        return blocks

    def __contains__(self, name: str) -> bool:
        return name in self._tools
