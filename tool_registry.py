from __future__ import annotations
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from tools import Tools
from agent_context import AgentContext, MAX_TOTAL_SUBAGENTS


def _make_write_tool(context: AgentContext) -> Callable:
    async def write_tool(text: str) -> str:
        async with context.lock:
            Tools.write_tool(text)
        return f'written: {text[:60]}'
    return write_tool


def _make_save_tool(context: AgentContext) -> Callable:
    async def save_tool(filename: str, content: str) -> str:
        async with context.lock:
            Tools.save_tool(filename, content)
        return f'saved to {filename}'
    return save_tool


def _make_read_file(context: AgentContext) -> Callable:
    async def read_file(filename: str) -> str:
        return Tools.read_file(filename)
    return read_file


def _make_list_files(context: AgentContext) -> Callable:
    async def list_files(directory: str = '.') -> str:
        return Tools.list_files(directory)
    return list_files


def _make_fetch_url(context: AgentContext) -> Callable:
    async def fetch_url(url: str) -> str:
        return Tools.fetch_url(url)
    return fetch_url


def _make_search_web(context: AgentContext) -> Callable:
    async def search_web(query: str) -> str:
        return Tools.search_web(query)
    return search_web


def _make_delete_file(context: AgentContext) -> Callable:
    async def delete_file(filename: str) -> str:
        return Tools.delete_file(filename)
    return delete_file


def _make_make_directory(context: AgentContext) -> Callable:
    async def make_directory(path: str) -> str:
        return Tools.make_directory(path)
    return make_directory


def _make_get_datetime(context: AgentContext) -> Callable:
    async def get_datetime() -> str:
        return Tools.get_datetime()
    return get_datetime


def _make_summarize_file(context: AgentContext) -> Callable:
    async def summarize_file(filename: str, max_chars: str = '2000') -> str:
        return Tools.summarize_file(filename, max_chars)
    return summarize_file


def _make_append_memory(context: AgentContext) -> Callable:
    async def append_memory(note: str) -> str:
        async with context.lock:
            return Tools.append_memory(note)
    return append_memory


def _make_recall_memory(context: AgentContext) -> Callable:
    async def recall_memory() -> str:
        return Tools.recall_memory()
    return recall_memory


def _make_search_and_fetch(context: AgentContext) -> Callable:
    async def search_and_fetch(query: str) -> str:
        return Tools.search_and_fetch(query)
    return search_and_fetch


def _make_multi_search(context: AgentContext) -> Callable:
    async def multi_search(queries_json: str) -> str:
        return Tools.multi_search(queries_json)
    return multi_search


def make_spawn_agent(context: AgentContext) -> Callable:
    async def spawn_agent(
        task: str,
        context_brief: str = '',
        tier: str = 'standard',
    ) -> str:
        model = context.resolve_model(tier)
        if model is None:
            msg = (
                '[spawn_agent] BLOCKED -- requested tier not confirmed by user. '
                'Use a lower tier or re-run and confirm when prompted.'
            )
            print(f'\n{msg}')
            return msg

        granted = await context.claim_spawn()
        if not granted:
            msg = (
                f'[spawn_agent] BLOCKED -- sub-agent cap reached '
                f'({context._spawned_count}/{MAX_TOTAL_SUBAGENTS}). '
                f'Handle directly: {task}'
            )
            print(f'\n{msg}')
            return msg

        spawn_num = context._spawned_count
        print(
            f'\n[spawn_agent] Launching sub-agent #{spawn_num} '
            f'(depth={context._current_depth + 1}, tier={tier}, model={model})'
            f'\n  Task: {task[:80]}'
            f'\n  Brief: {context_brief[:80] or "(none)"}'
        )

        from agent import run_agent

        child_ctx = context.child_context(new_brief=context_brief)
        sub_registry = ToolRegistry(context=child_ctx)
        allowed_tools = [t for t in sub_registry.names() if t != 'spawn_agent']

        async with context.semaphore:
            print(f'\n[sub-agent #{spawn_num}] Starting (model={model})')
            result = await run_agent(
                task=task,
                registry=sub_registry,
                model=model,
                max_iterations=5,
                tool_filter=allowed_tools,
                context=child_ctx,
            )
            print(f'\n[sub-agent #{spawn_num}] Done.')
            return result

    return spawn_agent


class ToolMeta(NamedTuple):
    func: Callable[..., Any]
    arg_names: List[str]
    description: str
    dangerous: bool = False


class ToolRegistry:
    def __init__(self, context: AgentContext | None = None) -> None:
        self._tools: Dict[str, ToolMeta] = {}
        self._context = context or AgentContext()
        self._register_defaults()

    def _register_defaults(self) -> None:
        ctx = self._context

        # --- File tools ---
        self.register_tool(
            'write_tool',
            _make_write_tool(ctx),
            arg_names=['text'],
            description='writes text to output.txt',
        )
        self.register_tool(
            'save_tool',
            _make_save_tool(ctx),
            arg_names=['filename', 'content'],
            description='saves content to a named file',
        )
        self.register_tool(
            'read_file',
            _make_read_file(ctx),
            arg_names=['filename'],
            description='reads and returns the full contents of a file',
        )
        self.register_tool(
            'list_files',
            _make_list_files(ctx),
            arg_names=['directory'],
            description='lists all files in a directory, defaults to current directory',
        )
        self.register_tool(
            'delete_file',
            _make_delete_file(ctx),
            arg_names=['filename'],
            description='deletes a file from disk',
            dangerous=True,
        )
        self.register_tool(
            'make_directory',
            _make_make_directory(ctx),
            arg_names=['path'],
            description='creates a directory and all intermediate directories',
        )
        self.register_tool(
            'summarize_file',
            _make_summarize_file(ctx),
            arg_names=['filename', 'max_chars'],
            description='reads a file and returns up to max_chars characters as a preview',
        )

        # --- Utility tools ---
        self.register_tool(
            'get_datetime',
            _make_get_datetime(ctx),
            arg_names=[],
            description='returns the current date and time as a string',
        )
        self.register_tool(
            'append_memory',
            _make_append_memory(ctx),
            arg_names=['note'],
            description='appends a timestamped note to memory.md for cross-session recall',
        )
        self.register_tool(
            'recall_memory',
            _make_recall_memory(ctx),
            arg_names=[],
            description='reads and returns all stored notes from memory.md',
        )

        # --- Web and research tools ---
        self.register_tool(
            'fetch_url',
            _make_fetch_url(ctx),
            arg_names=['url'],
            description='fetches and returns the text content of a URL (first 8000 chars)',
        )
        self.register_tool(
            'search_web',
            _make_search_web(ctx),
            arg_names=['query'],
            description='searches the web via DuckDuckGo and returns results with source URLs',
        )
        self.register_tool(
            'search_and_fetch',
            _make_search_and_fetch(ctx),
            arg_names=['query'],
            description='searches the web and auto-fetches the top result URL for full content',
        )
        self.register_tool(
            'multi_search',
            _make_multi_search(ctx),
            arg_names=['queries_json'],
            description='runs up to 5 parallel search_web queries from a JSON array and returns combined results',
        )

        # --- Agent tools ---
        self.register_tool(
            'spawn_agent',
            make_spawn_agent(ctx),
            arg_names=['task', 'context_brief', 'tier'],
            description="delegates a sub-task to a sub-agent; tier is 'light', 'standard', or 'heavy'",
        )

    def register_tool(
        self,
        name: str,
        func: Callable,
        *,
        arg_names: Optional[List[str]] = None,
        description: str = '',
        dangerous: bool = False,
    ) -> None:
        self._tools[name] = ToolMeta(
            func=func,
            arg_names=arg_names or ['arg'],
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
            inner = '\n'.join(
                f'  <arg{i+1}>{label}</arg{i+1}>'
                for i, label in enumerate(meta.arg_names)
            )
            danger_note = '  [DANGEROUS -- state your intent before calling]' if meta.dangerous else ''
            blocks.append(
                f'  {name} -- {meta.description}{danger_note}\n'
                f'  <{name}>\n{inner}\n  </{name}>'
            )
        return blocks

    def __contains__(self, name: str) -> bool:
        return name in self._tools
