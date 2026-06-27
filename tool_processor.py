import re
import asyncio
import inspect
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
    registered tool is automatically recognised — no changes needed here.
    Single-argument tools still work — they just use <arg1> only.
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

    def feed(self, chunk: Any) -> None:
        text = getattr(chunk.message, "content", None) or ""
        if text:
            self._buffer += text
            self._flush_complete_blocks()

    async def finalize(self) -> None:
        for name in self._registry.names():
            open_tag = f"<{name}>"
            close_tag = f"</{name}>"
            if open_tag in self._buffer and close_tag not in self._buffer:
                self._buffer += f"\n{close_tag}"
        await self._flush_complete_blocks_async()
        self._buffer = ""

    def flush_results(self) -> List[Dict[str, str]]:
        results = list(self._results)
        self._results.clear()
        return results

    def rebuild_pattern(self) -> None:
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
            return [inner.strip()]

        ordered = sorted(matches, key=lambda m: int(m[0]))
        return [value for _, value in ordered]

    def _flush_complete_blocks(self) -> None:
        while True:
            match = self._pattern.search(self._buffer)
            if not match:
                break
            tool_name = match.group(1)
            inner     = match.group(2)
            func = self._registry.get(tool_name)
            if func is not None and not inspect.iscoroutinefunction(func):
                self._dispatch_sync(tool_name, inner)
            self._buffer = self._buffer[:match.start()] + self._buffer[match.end():]

    async def _flush_complete_blocks_async(self) -> None:
        while True:
            match = self._pattern.search(self._buffer)
            if not match:
                break
            tool_name = match.group(1)
            inner     = match.group(2)
            await self._dispatch(tool_name, inner)
            self._buffer = self._buffer[:match.start()] + self._buffer[match.end():]

    def _dispatch_sync(self, tool_name: str, inner: str) -> None:
        if tool_name not in self._registry:
            print(f"[ToolProcessor] Unknown tool: {tool_name!r} — skipping")
            return

        args = self._parse_args(inner)
        args = [
            arg.replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'")
            for arg in args
        ]

        func = self._registry.get(tool_name)
        try:
            result = func(*args)
            if result is not None:
                self._results.append({"tool": tool_name, "result": str(result)})
        except TypeError as e:
            print(f"[ToolProcessor] Argument mismatch for {tool_name!r}: {e}")

    async def _dispatch(self, tool_name: str, inner: str) -> None:
        if tool_name not in self._registry:
            print(f"[ToolProcessor] Unknown tool: {tool_name!r} — skipping")
            return

        args = self._parse_args(inner)
        args = [
            arg.replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'")
            for arg in args
        ]

        func = self._registry.get(tool_name)
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*args)
            else:
                result = func(*args)
            if result is not None:
                self._results.append({"tool": tool_name, "result": str(result)})
        except TypeError as e:
            print(f"[ToolProcessor] Argument mismatch for {tool_name!r}: {e}")
