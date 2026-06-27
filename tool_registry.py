from __future__ import annotations
from typing import Any, Callable, Dict, List, NamedTuple
from tools import Tools


class ToolMeta(NamedTuple):
    func: Callable[..., Any]
    arg_names: List[str]
    description: str


def _make_write_tool() -> Callable:
    async def write_tool(text: str) -> str:
        Tools.write_tool(text)
        return f'written: {text[:60]}'
    return write_tool


def _make_save_tool() -> Callable:
    async def save_tool(filename: str, content: str) -> str:
        Tools.save_tool(filename, content)
        return f'saved to {filename}'
    return save_tool


def _make_read_file() -> Callable:
    async def read_file(filename: str) -> str:
        return Tools.read_file(filename)
    return read_file


def _make_list_files() -> Callable:
    async def list_files(directory: str = '.') -> str:
        return Tools.list_files(directory)
    return list_files


def _make_fetch_url() -> Callable:
    async def fetch_url(url: str) -> str:
        return Tools.fetch_url(url)
    return fetch_url


def _make_search_web() -> Callable:
    async def search_web(query: str) -> str:
        return Tools.search_web(query)
    return search_web


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolMeta] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register_tool(
            'write_tool',
            _make_write_tool(),
            arg_names=['text'],
            description='appends text to output.txt in the current directory',
        )
        self.register_tool(
            'save_tool',
            _make_save_tool(),
            arg_names=['filename', 'content'],
            description='appends content to a named file',
        )
        self.register_tool(
            'read_file',
            _make_read_file(),
            arg_names=['filename'],
            description='reads and returns the full contents of a file',
        )
        self.register_tool(
            'list_files',
            _make_list_files(),
            arg_names=['directory'],
            description='lists all entries in a directory, defaults to current directory',
        )
        self.register_tool(
            'fetch_url',
            _make_fetch_url(),
            arg_names=['url'],
            description='fetches and returns the first 4000 characters of a URL',
        )
        self.register_tool(
            'search_web',
            _make_search_web(),
            arg_names=['query'],
            description='searches the web via DuckDuckGo and returns top 5 results',
        )

    def register_tool(self, name, func, *, arg_names=None, description='') -> None:
        self._tools[name] = ToolMeta(
            func=func,
            arg_names=arg_names or ['arg'],
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
        blocks = []
        for name, meta in self._tools.items():
            inner = '\n'.join(
                f'  <arg{i+1}>{label}</arg{i+1}>'
                for i, label in enumerate(meta.arg_names)
            )
            blocks.append(
                f'  {name} -- {meta.description}\n'
                f'  <{name}>\n{inner}\n  </{name}>'
            )
        return blocks

    def __contains__(self, name: str) -> bool:
        return name in self._tools
