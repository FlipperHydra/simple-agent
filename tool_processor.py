import re
import asyncio
from typing import Any, Callable, Dict, List, Optional

from tool_registry import ToolRegistry
from prompts import soul_edit_proposal_display, soul_remove_proposal_display


class ToolProcessor:
    """
    Parses tool calls in the format:

        <tool_name>
        <arg1>value one</arg1>
        <arg2>value two</arg2>
        </tool_name>

    Tools flagged as dangerous=True prompt the user for confirmation.
    propose_soul_edit and propose_soul_remove are intercepted specially.
    """

    _ARG_PATTERN = re.compile(
        r"<arg(\d+)>\s*(.*?)\s*</arg\1>",
        flags=re.DOTALL,
    )

    def __init__(
        self,
        registry: ToolRegistry,
        soul_writer: Optional[Callable[[str, str], None]] = None,
        soul_reader: Optional[Callable[[str], str]] = None,
        soul_remover: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._registry = registry
        self._soul_writer = soul_writer
        self._soul_reader = soul_reader
        self._soul_remover = soul_remover
        self._buffer: str = ''
        self._pattern: re.Pattern = self._build_pattern()
        self._results: List[Dict[str, str]] = []
        self._pending: List[tuple] = []

    def feed(self, chunk: Any) -> None:
        text = getattr(chunk.message, 'content', None) or ''
        if text:
            self._buffer += text
            self._collect_complete_blocks()

    async def finalize(self) -> None:
        for name in self._registry.names():
            open_tag = f'<{name}>'
            close_tag = f'</{name}>'
            if open_tag in self._buffer and close_tag not in self._buffer:
                self._buffer += f'\n{close_tag}'
        self._collect_complete_blocks()
        self._buffer = ''
        for tool_name, inner in self._pending:
            await self._dispatch(tool_name, inner)
        self._pending.clear()

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
        name_alts = '|'.join(re.escape(n) for n in names)
        pattern = (
            r'<({names})>'
            r'\s*(.*?)\s*'
            r'</\1>'
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
            self._pending.append((match.group(1), match.group(2)))
            self._buffer = self._buffer[:match.start()] + self._buffer[match.end():]

    async def _confirm_dangerous(self, tool_name: str, args: List[str]) -> bool:
        args_preview = ', '.join(f'"{a[:40]}"' for a in args) if args else ''
        prompt = f'\n[!] DANGEROUS: {tool_name}({args_preview}) -- confirm? [y/N]: '
        answer = await asyncio.to_thread(input, prompt)
        return answer.strip().lower() in ('y', 'yes')

    async def _handle_soul_edit_proposal(self, section: str, proposed_content: str) -> None:
        existing = ''
        if self._soul_reader is not None:
            existing = self._soul_reader(section)

        prompt = soul_edit_proposal_display(section, proposed_content, existing)
        answer = await asyncio.to_thread(input, prompt)

        if answer.strip().lower() in ('y', 'yes'):
            if self._soul_writer is not None:
                self._soul_writer(section, proposed_content)
                self._results.append({
                    'tool': 'propose_soul_edit',
                    'result': f"[soul_edit] Section '{section}' accepted and updated."
                })
            else:
                self._results.append({
                    'tool': 'propose_soul_edit',
                    'result': '[soul_edit] No soul writer configured.'
                })
        else:
            self._results.append({
                'tool': 'propose_soul_edit',
                'result': '[soul_edit] rejected by user'
            })

    async def _handle_soul_remove_proposal(self, section: str) -> None:
        existing = ''
        if self._soul_reader is not None:
            existing = self._soul_reader(section)

        prompt = soul_remove_proposal_display(section, existing)
        answer = await asyncio.to_thread(input, prompt)

        if answer.strip().lower() in ('y', 'yes'):
            if self._soul_remover is not None:
                self._soul_remover(section)
                self._results.append({
                    'tool': 'propose_soul_remove',
                    'result': f"[soul_remove] Section '{section}' removed."
                })
            else:
                self._results.append({
                    'tool': 'propose_soul_remove',
                    'result': '[soul_remove] No soul remover configured.'
                })
        else:
            self._results.append({
                'tool': 'propose_soul_remove',
                'result': '[soul_remove] rejected by user'
            })

    async def _dispatch(self, tool_name: str, inner: str) -> None:
        if tool_name not in self._registry and tool_name not in ('propose_soul_edit', 'propose_soul_remove'):
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

        if tool_name == 'propose_soul_edit':
            if len(args) != 2:
                print('[ToolProcessor] propose_soul_edit requires 2 arguments')
                return
            await self._handle_soul_edit_proposal(args[0], args[1])
            return

        if tool_name == 'propose_soul_remove':
            if len(args) != 1:
                print('[ToolProcessor] propose_soul_remove requires 1 argument')
                return
            await self._handle_soul_remove_proposal(args[0])
            return

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
                self._results.append({'tool': tool_name, 'result': str(result)})
        except TypeError as e:
            print(f"[ToolProcessor] Argument mismatch for {tool_name!r}: {e}")
