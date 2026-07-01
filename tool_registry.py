from __future__ import annotations
from typing import Any, Callable, Dict, List, NamedTuple
from tools import Tools


class ToolMeta(NamedTuple):
    func: Callable[..., Any]
    arg_names: List[str]
    description: str
    dangerous: bool = False


def _make_write_tool() -> Callable:
    async def write_tool(text: str) -> str:
        Tools.write_tool(text)
        return f'written: {text[:60]}'
    return write_tool


def _make_save_tool() -> Callable:
    async def save_tool(filename: str, content: str) -> str:
        return Tools.save_tool(filename, content)
    return save_tool


def _make_overwrite_file() -> Callable:
    async def overwrite_file(filename: str, content: str) -> str:
        return Tools.overwrite_file(filename, content)
    return overwrite_file


def _make_get_datetime() -> Callable:
    async def get_datetime() -> str:
        return Tools.get_datetime()
    return get_datetime


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


def _make_delete_file() -> Callable:
    async def delete_file(filename: str) -> str:
        return Tools.delete_file(filename)
    return delete_file


def _make_make_directory() -> Callable:
    async def make_directory(path: str) -> str:
        return Tools.make_directory(path)
    return make_directory


def _make_copy_file() -> Callable:
    async def copy_file(src: str, dest: str) -> str:
        return Tools.copy_file(src, dest)
    return copy_file


def _make_move_file() -> Callable:
    async def move_file(src: str, dest: str) -> str:
        return Tools.move_file(src, dest)
    return move_file


def _make_append_memory() -> Callable:
    async def append_memory(note: str) -> str:
        return Tools.append_memory(note)
    return append_memory


def _make_recall_memory() -> Callable:
    async def recall_memory() -> str:
        return Tools.recall_memory()
    return recall_memory


def _make_clear_memory() -> Callable:
    async def clear_memory() -> str:
        return Tools.clear_memory()
    return clear_memory


def _make_write_json() -> Callable:
    async def write_json(filename: str, key: str, value: str) -> str:
        return Tools.write_json(filename, key, value)
    return write_json


def _make_read_json() -> Callable:
    async def read_json(filename: str, key: str = '') -> str:
        return Tools.read_json(filename, key)
    return read_json


def _make_eval_math() -> Callable:
    async def eval_math(expression: str) -> str:
        return Tools.eval_math(expression)
    return eval_math


def _make_summarize_file() -> Callable:
    async def summarize_file(filename: str) -> str:
        return Tools.summarize_file(filename)
    return summarize_file


def _make_summarize_file_chunk() -> Callable:
    async def summarize_file_chunk(filename: str, chunk_index: str, chunk_summary: str) -> str:
        return Tools.summarize_file_chunk(filename, int(chunk_index), chunk_summary)
    return summarize_file_chunk


def _make_summarize_file_finalize() -> Callable:
    async def summarize_file_finalize(filename: str) -> str:
        return Tools.summarize_file_finalize(filename)
    return summarize_file_finalize


def _make_zip_files() -> Callable:
    async def zip_files(filenames_csv: str, output_zip: str) -> str:
        return Tools.zip_files(filenames_csv, output_zip)
    return zip_files


def _make_propose_soul_edit() -> Callable:
    async def propose_soul_edit(section: str, proposed_content: str) -> str:
        return Tools.propose_soul_edit(section, proposed_content)
    return propose_soul_edit


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
            'overwrite_file',
            _make_overwrite_file(),
            arg_names=['filename', 'content'],
            description='replaces the entire contents of a file with new content',
        )
        self.register_tool(
            'get_datetime',
            _make_get_datetime(),
            arg_names=[],
            description='returns the current date and time as a formatted string',
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
            description='searches the web via DDGS and returns top 5 results',
        )
        self.register_tool(
            'delete_file',
            _make_delete_file(),
            arg_names=['filename'],
            description='permanently deletes a file from disk [DANGEROUS]',
            dangerous=True,
        )
        self.register_tool(
            'make_directory',
            _make_make_directory(),
            arg_names=['path'],
            description='creates a directory and any missing parent directories',
        )
        self.register_tool(
            'copy_file',
            _make_copy_file(),
            arg_names=['src', 'dest'],
            description='copies a file from src to dest, leaving the original intact',
        )
        self.register_tool(
            'move_file',
            _make_move_file(),
            arg_names=['src', 'dest'],
            description='moves or renames a file from src to dest [DANGEROUS]',
            dangerous=True,
        )
        self.register_tool(
            'append_memory',
            _make_append_memory(),
            arg_names=['note'],
            description='appends a timestamped note to memory.json log for persistent recall',
        )
        self.register_tool(
            'recall_memory',
            _make_recall_memory(),
            arg_names=[],
            description='reads and returns all facts and log entries from memory.json',
        )
        self.register_tool(
            'clear_memory',
            _make_clear_memory(),
            arg_names=[],
            description='wipes all entries from memory.json [DANGEROUS]',
            dangerous=True,
        )
        self.register_tool(
            'write_json',
            _make_write_json(),
            arg_names=['filename', 'key', 'value'],
            description='upserts a key-value pair into a JSON file',
        )
        self.register_tool(
            'read_json',
            _make_read_json(),
            arg_names=['filename', 'key'],
            description='reads a key from a JSON file, or returns the full file if key is omitted',
        )
        self.register_tool(
            'eval_math',
            _make_eval_math(),
            arg_names=['expression'],
            description='safely evaluates a math expression using AST whitelist (no arbitrary code)',
        )
        self.register_tool(
            'summarize_file',
            _make_summarize_file(),
            arg_names=['filename'],
            description='init: reads file, calculates chunks, returns dispatch instructions for iterative summarization',
        )
        self.register_tool(
            'summarize_file_chunk',
            _make_summarize_file_chunk(),
            arg_names=['filename', 'chunk_index', 'chunk_summary'],
            description='per-chunk step: stores agent summary and anchor notes for one chunk; returns next chunk content',
        )
        self.register_tool(
            'summarize_file_finalize',
            _make_summarize_file_finalize(),
            arg_names=['filename'],
            description='final step: synthesizes all chunk summaries and anchor notes into a complete file summary',
        )
        self.register_tool(
            'zip_files',
            _make_zip_files(),
            arg_names=['filenames_csv', 'output_zip'],
            description='compresses a comma-separated list of files into a zip archive',
        )
        self.register_tool(
            'propose_soul_edit',
            _make_propose_soul_edit(),
            arg_names=['section', 'proposed_content'],
            description='proposes an update to a soul.md section for user approval',
        )

    def register_tool(
        self,
        name: str,
        func: Callable,
        *,
        arg_names: List[str] = None,
        description: str = '',
        dangerous: bool = False,
    ) -> None:
        self._tools[name] = ToolMeta(
            func=func,
            arg_names=arg_names or [],
            description=description,
            dangerous=dangerous,
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
            danger_tag = '  [DANGEROUS]' if meta.dangerous else ''
            if meta.arg_names:
                inner = '\n'.join(
                    f'  <arg{i+1}>{label}</arg{i+1}>'
                    for i, label in enumerate(meta.arg_names)
                )
                block = (
                    f'  {name} -- {meta.description}{danger_tag}\n'
                    f'  <{name}>\n{inner}\n  </{name}>'
                )
            else:
                block = (
                    f'  {name} -- {meta.description}{danger_tag}\n'
                    f'  <{name}></{name}>'
                )
            blocks.append(block)
        return blocks

    def __contains__(self, name: str) -> bool:
        return name in self._tools
