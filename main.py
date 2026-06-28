from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime
import ollama
from tool_registry import ToolRegistry
from tool_processor import ToolProcessor
from prompts import (
    tool_prompt,
    FORMAT_PROMPT,
    REASONING_PROMPT,
    MEMORY_PROMPT,
    soul_prompt,
    soul_update_prompt,
)

_client = ollama.AsyncClient()

SOUL_FILE = 'soul.md'
MEMORY_FILE = 'memory.md'


def _load_soul() -> str | None:
    if os.path.exists(SOUL_FILE):
        with open(SOUL_FILE, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None


def _load_memory() -> str | None:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content if content else None
    return None


def _build_system_messages(registry: ToolRegistry) -> list:
    messages = [
        {'role': 'system', 'content': tool_prompt(registry)},
        {'role': 'system', 'content': FORMAT_PROMPT},
        {'role': 'system', 'content': REASONING_PROMPT},
        {'role': 'system', 'content': MEMORY_PROMPT},
    ]
    soul = _load_soul()
    if soul:
        messages.append({'role': 'system', 'content': soul_prompt(soul)})
    return messages


async def _stream_response(client, model: str, messages: list, registry: ToolRegistry):
    """Stream a response, print thinking + answer, return (full_response, tool_results)."""
    tp = ToolProcessor(registry)
    response = await client.chat(
        model=model,
        messages=messages,
        think=True,
        stream=True,
    )

    full_response = ''
    in_thinking = False

    async for chunk in response:
        if chunk.message.thinking:
            if not in_thinking:
                print('\n-- Thinking --------------------------------------------------\n')
                in_thinking = True
            print(chunk.message.thinking, end='', flush=True)
        elif chunk.message.content:
            if in_thinking:
                print('\n\n-- Final Answer -----------------------------------------------\n')
                in_thinking = False
            print(chunk.message.content, end='', flush=True)
            full_response += chunk.message.content
            tp.feed(chunk)

    await tp.finalize()
    print()
    return full_response, tp.flush_results()


async def _run_soul_update(client, model: str, registry: ToolRegistry) -> None:
    """Run a one-shot soul update and write the result back to soul.md."""
    soul_content = _load_soul() or '(soul.md not found)'
    memory_content = _load_memory() or '(memory.md is empty)'

    prompt = soul_update_prompt(memory_content, soul_content)
    update_messages = [
        {'role': 'system', 'content': tool_prompt(registry)},
        {'role': 'user', 'content': prompt},
    ]

    print('\n-- Soul Update ------------------------------------------------\n')
    response = await client.chat(
        model=model,
        messages=update_messages,
        think=True,
        stream=True,
    )

    full = ''
    in_thinking = False
    async for chunk in response:
        if chunk.message.thinking:
            if not in_thinking:
                print('Thinking...\n')
                in_thinking = True
            print(chunk.message.thinking, end='', flush=True)
        elif chunk.message.content:
            if in_thinking:
                print('\n\n-- Insights + Updated soul.md --------------------------------\n')
                in_thinking = False
            print(chunk.message.content, end='', flush=True)
            full += chunk.message.content

    print()

    # Extract updated soul.md content (between first # heading and end)
    soul_start = full.find('# ')
    if soul_start != -1:
        updated_soul = full[soul_start:].strip()
        with open(SOUL_FILE, 'w', encoding='utf-8') as f:
            f.write(updated_soul + '\n')
        print(f'\n[soul_update] soul.md updated.')
    else:
        print('\n[soul_update] Could not extract updated soul.md -- no changes written.')


async def main() -> None:
    registry = ToolRegistry()
    current_model = 'gemma4'

    system_messages = _build_system_messages(registry)
    messages = list(system_messages)

    print('Agent ready.')
    print('Commands: /tools, /save_session, /load_session <file>, /model <name>, /soul_update, /clear, /quit')

    while True:
        user_message = input('\nYou: ').strip()

        if not user_message:
            continue

        # --- REPL commands ---

        if user_message == '/quit':
            break

        if user_message == '/clear':
            system_messages = _build_system_messages(registry)
            messages = list(system_messages)
            print('[History cleared]')
            continue

        if user_message == '/tools':
            print('\n-- Registered Tools -------------------------------------------')
            for name, meta in registry.all().items():
                danger = '  [DANGEROUS]' if meta.dangerous else ''
                args = ', '.join(meta.arg_names) if meta.arg_names else 'no args'
                print(f'  {name}({args}) -- {meta.description}{danger}')
            print()
            continue

        if user_message == '/save_session':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'session_{timestamp}.json'
            # Exclude system messages from the saved file
            saveable = [m for m in messages if m['role'] != 'system']
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(saveable, f, indent=2, ensure_ascii=False)
            print(f'[save_session] Saved {len(saveable)} messages to {filename}')
            continue

        if user_message.startswith('/load_session'):
            parts = user_message.split(maxsplit=1)
            if len(parts) < 2:
                print('[load_session] Usage: /load_session <filename>')
                continue
            filepath = parts[1].strip()
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                messages = list(system_messages) + loaded
                print(f'[load_session] Loaded {len(loaded)} messages from {filepath}')
            except FileNotFoundError:
                print(f'[load_session] File not found: {filepath}')
            except json.JSONDecodeError as e:
                print(f'[load_session] Invalid JSON: {e}')
            continue

        if user_message.startswith('/model'):
            parts = user_message.split(maxsplit=1)
            if len(parts) < 2:
                print(f'[model] Current model: {current_model}')
            else:
                current_model = parts[1].strip()
                print(f'[model] Switched to: {current_model}')
            continue

        if user_message == '/soul_update':
            await _run_soul_update(_client, current_model, registry)
            continue

        # --- Normal chat turn ---

        messages.append({'role': 'user', 'content': user_message})

        full_response, tool_results = await _stream_response(
            _client, current_model, messages, registry
        )
        messages.append({'role': 'assistant', 'content': full_response})

        for tr in tool_results:
            messages.append({
                'role': 'user',
                'content': f"[Tool result: {tr['tool']}]\n{tr['result']}"
            })

        if tool_results:
            followup_response, _ = await _stream_response(
                _client, current_model, messages, registry
            )
            messages.append({'role': 'assistant', 'content': followup_response})

    print('\n[Session complete]')


if __name__ == '__main__':
    asyncio.run(main())
