import re
import asyncio
from typing import Any, Dict, List

from tool_registry import ToolRegistry


class ToolProcessor:
    """
    Parses tool calls in the format:

        <tool_name>
        <arg1>value one</arg1>
        <arg2>value two</arg2>
        </tool_name>

    Tag names are derived dynamically from the registry, so any newly
    registered tool is automatically recognised -- no changes needed here.
    Single-argument tools still work -- they just use <arg1> only.
    Tools flagged as dangerous=True will prompt the user for confirmation
    before executing.
    """

    _ARG_PATTERN = re.compile(
        r"<arg(\d+)>\s*(.*?)\s*</arg\1>",
        flags=re.DOTALL,
    )

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry
        self._buffer: str = ""
        self._pattern: re.Pattern = self._build_pattern()
        self._results: List[Dict[str, str]] = []
        self._pending: List[tuple] = []

    def feed(self, chunk: Any) -> None:
        """Feed a stream chunk into the buffer and collect complete tool blocks."""
        text = getattr(chunk.message, "content", None) or ""
        if text:
            self._buffer += text
            self._collect_complete_blocks()

    async def finalize(self) -> None:
        """
        Called after the stream ends. Closes any orphaned open tool tag
        so its content is not silently dropped, then dispatches all
        pending tool calls.
        """
        for name in self._registry.names():
            open_tag = f"<{name}>"
            close_tag = f"</{name}>"
            if open_tag in self._buffer and close_tag not in self._buffer:
                self._buffer += f"\n{close_tag}"
        self._collect_complete_blocks()
        self._buffer = ""

        for tool_name, inner in self._pending:
            await self._dispatch(tool_name, inner)
        self._pending.clear()

    def flush_results(self) -> List[Dict[str, str]]:
        results = list(self._results)
        self._results.clear()
        return results

    def rebuild_pattern(self) -> None:
        """Call this if you register new tools after __init__."""
        self._pattern = self._build_pattern()

    def _build_pattern(self) -> re.Pattern:
        names = self._registry.names()
        if not names:
            return re.compile(r'(?!)')

        name_alts = "|".join(re.escape(n) for n in names)
        pattern = (
            r"<({names})>"
            r"\s*(.*?)\s*"
            r"</\1>"
        ).format(names=name_alts)
        return re.compile(pattern, flags=re.DOTALL)

    def _parse_args(self, inner: str) -> List[str]:
        matches = self._ARG_PATTERN.findall(inner)
        if not matches:
            stripped = inner.strip()
            return [stripped] if stripped else []
        ordered = sorted(matches, key=lambda m: int(m[0]))
        return [value for _, value in ordered]

    def _collect_complete_blocks(self) -> None:
        while True:
            match = self._pattern.search(self._buffer)
            if not match:
                break
            tool_name = match.group(1)
            inner     = match.group(2)
            self._pending.append((tool_name, inner))
            self._buffer = self._buffer[:match.start()] + self._buffer[match.end():]

    async def _confirm_dangerous(self, tool_name: str, args: List[str]) -> bool:
        """Prompt the user to confirm a dangerous tool call. Non-blocking via executor."""
        args_preview = ', '.join(f'"{a[:40]}"' for a in args) if args else ''
        prompt = f'\n[!] DANGEROUS: {tool_name}({args_preview}) -- confirm? [y/N]: '
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, input, prompt)
        return answer.strip().lower() in ('y', 'yes')

    async def _dispatch(self, tool_name: str, inner: str) -> None:
        """Await the async tool function and store its result."""
        if tool_name not in self._registry:
            print(f"[ToolProcessor] Unknown tool: {tool_name!r} -- skipping")
            return

        args = self._parse_args(inner)

        args = [
            arg
            .replace('\\n', '\n')
            .replace('\\t', '\t')
            .replace("\\'", "'")
            for arg in args
        ]

        meta = self._registry.meta(tool_name)
        if meta and meta.dangerous:
            confirmed = await self._confirm_dangerous(tool_name, args)
            if not confirmed:
                print(f'[ToolProcessor] {tool_name} cancelled by user.')
                self._results.append({
                    'tool': tool_name,
                    'result': '[cancelled by user -- dangerous tool was not executed]'
                })
                return

        func = self._registry.get(tool_name)
        try:
            result = func(*args)
            if asyncio.iscoroutine(result):
                result = await result
            if result is not None:
                self._results.append({
                    'tool': tool_name,
                    'result': str(result)
                })
        except TypeError as e:
            print(f"[ToolProcessor] Argument mismatch for {tool_name!r}: {e}")
