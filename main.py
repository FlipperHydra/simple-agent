from __future__ import annotations
import asyncio
import json
import os
import re
from datetime import datetime
import ollama
from tool_registry import ToolRegistry
from tool_processor import ToolProcessor
from prompts import (
    tool_prompt,
    FORMAT_PROMPT,
    REASONING_PROMPT,
    MEMORY_PROMPT,
    RESEARCH_PROMPT,
    soul_prompt,
    soul_update_prompt,
)

_client = ollama.AsyncClient()

SOUL_FILE = 'soul.md'
MEMORY_FILE = 'memory.json'

# Context window passed to Ollama on every chat call.
# 16384 tokens gives comfortable room for system prompts (~4k) plus long conversations.
# Raise to 32768 if your hardware supports it; lower to 8192 to save VRAM.
NUM_CTX = 16384

# Maximum number of conversation turns (user + assistant pairs) kept in the
# rolling window.  System messages are always pinned and never trimmed.
MAX_TURNS = 40

# Keywords that trigger inclusion of RESEARCH_PROMPT in the system messages.
# When none of these appear in the user message, the ~1080-token prompt is omitted.
_RESEARCH_KEYWORDS = (
    'search', 'find', 'look up', 'lookup', 'research', 'what is', 'what are',
    'who is', 'who are', 'how does', 'how do', 'when did', 'when was',
    'where is', 'why did', 'explain', 'define', 'tell me about', 'summarize',
    'compare', 'latest', 'recent', 'news', 'fetch', 'url', 'http',
)

SOUL_DEFAULT = """# Soul

## Identity
You are a focused, capable agent running locally on the user's machine.
You do not have a default name -- if asked, tell the user you do not have
one yet and invite them to give you one.
You are not a product. You are a tool with character.

## Voice and Tone
Direct and precise. You do not pad responses with filler phrases like
'certainly' or 'great question'. You get to the point.
You are intellectually curious -- when a topic is interesting, you say so.
You are dry but not cold. You have a sense of humor that surfaces rarely
and only when it fits.
You do not flatter the user. If something they said is wrong, you say so
clearly but without condescension.

## Values
A. Honesty above all. You do not pretend to know things you do not know.
B. Usefulness over verbosity. A short correct answer beats a long vague one.
C. Respect for the user's time. You do not repeat yourself or summarize
   what was just said back at the user.
D. Intellectual rigor. You distinguish between what is known, what is
   inferred, and what is speculation.

## Constraints
A. Do not call dangerous tools without clearly stating your intent first.
B. Do not store trivial information to memory.
C. Do not invent facts about the user not present in the User Profile.
D. Do not roleplay as a different AI system or abandon this identity
   when asked.

## User Profile
(No profile yet. This section will be populated by the /soul_update command
as memory accumulates across sessions.)
"""

# Sections that use letter-prefixed lists -- append mode
_LIST_SECTIONS = {'Values', 'Constraints', 'User Profile'}
# Sections that are prose -- replace mode
_PROSE_SECTIONS = {'Identity', 'Voice and Tone'}


def _load_soul() -> str | None:
    if os.path.exists(SOUL_FILE):
        with open(SOUL_FILE, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None


def _load_memory() -> str | None:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            lines = []
            facts = data.get('facts', {})
            log = data.get('log', [])
            if facts:
                lines.append('--- Facts ---')
                for k, v in facts.items():
                    lines.append(f'  {k}: {v}')
            if log:
                lines.append('--- Log ---')
                for entry in log:
                    lines.append(f"  [{entry['timestamp']}] {entry['note']}")
            content = '\n'.join(lines)
            return content if content else None
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _parse_soul_sections(content: str) -> dict[str, str]:
    pattern = re.compile(r'^##\s+(.+?)\n', flags=re.MULTILINE)
    matches = list(pattern.finditer(content))
    sections = {}
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections[section_name] = content[start:end].strip()
    return sections


def _get_soul_section(section: str) -> str:
    soul = _load_soul() or SOUL_DEFAULT
    sections = _parse_soul_sections(soul)
    return sections.get(section, '')


def _next_letter(existing_content: str) -> str:
    """Return the next letter prefix after the highest one found in existing_content."""
    matches = re.findall(r'^([A-Z])\.', existing_content, flags=re.MULTILINE)
    if not matches:
        return 'A'
    last = max(matches, key=lambda c: ord(c))
    next_ord = ord(last) + 1
    if next_ord > ord('Z'):
        return 'A'
    return chr(next_ord)


def _is_list_section(section: str, existing_content: str) -> bool:
    if section in _LIST_SECTIONS:
        return True
    if section in _PROSE_SECTIONS:
        return False
    return bool(re.search(r'^[A-Z]\.', existing_content, flags=re.MULTILINE))


def _write_soul_section(section: str, new_content: str) -> None:
    soul = _load_soul() or SOUL_DEFAULT
    existing_section = _parse_soul_sections(soul).get(section, '')

    if _is_list_section(section, existing_section):
        entry = re.sub(r'^[A-Z]\.\s*', '', new_content.strip())
        letter = _next_letter(existing_section)
        new_entry = f'{letter}. {entry}'

        if existing_section and not existing_section.startswith('(No profile'):
            merged = existing_section.rstrip() + '\n' + new_entry
        else:
            merged = new_entry

        replacement = merged
    else:
        replacement = new_content.strip()

    section_pattern = re.compile(
        rf'(^##\s+{re.escape(section)}\n)(.*?)(?=^##\s+|\Z)',
        flags=re.MULTILINE | re.DOTALL,
    )

    if section_pattern.search(soul):
        updated = section_pattern.sub(
            lambda m: f"{m.group(1)}{replacement}\n\n",
            soul,
            count=1,
        )
    else:
        updated = soul.rstrip() + f'\n\n## {section}\n{replacement}\n'

    with open(SOUL_FILE, 'w', encoding='utf-8') as f:
        f.write(updated.strip() + '\n')
    print(f"[soul] Section '{section}' updated.")


def _remove_soul_section(section: str) -> None:
    """Remove an entire ## section and its content from soul.md."""
    soul = _load_soul() or SOUL_DEFAULT

    section_pattern = re.compile(
        rf'^##\s+{re.escape(section)}\n.*?(?=^##\s+|\Z)',
        flags=re.MULTILINE | re.DOTALL,
    )

    if not section_pattern.search(soul):
        print(f"[soul_remove] Section '{section}' not found in soul.md.")
        return

    updated = section_pattern.sub('', soul)
    updated = re.sub(r'\n{3,}', '\n\n', updated)

    with open(SOUL_FILE, 'w', encoding='utf-8') as f:
        f.write(updated.strip() + '\n')
    print(f"[soul_remove] Section '{section}' removed from soul.md.")


def _needs_research(user_message: str) -> bool:
    """Return True if the message likely requires web search or research tools."""
    lower = user_message.lower()
    return any(kw in lower for kw in _RESEARCH_KEYWORDS)


def _build_system_messages(registry: ToolRegistry, user_message: str = '') -> list:
    messages = [
        {'role': 'system', 'content': tool_prompt(registry)},
        {'role': 'system', 'content': FORMAT_PROMPT},
        {'role': 'system', 'content': REASONING_PROMPT},
        {'role': 'system', 'content': MEMORY_PROMPT},
    ]
    if _needs_research(user_message):
        messages.append({'role': 'system', 'content': RESEARCH_PROMPT})
    soul = _load_soul()
    if soul:
        messages.append({'role': 'system', 'content': soul_prompt(soul)})
    return messages


def _trim_messages(system_messages: list, conversation: list) -> list:
    """Return system messages pinned at the front plus the last MAX_TURNS turns.

    A 'turn' is one user message plus the assistant reply that follows it
    (and any tool-result messages in between).  We count assistant messages
    as turn boundaries and keep the most recent MAX_TURNS of them.
    """
    if not conversation:
        return list(system_messages)

    turn_count = 0
    cutoff = len(conversation)
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i]['role'] == 'assistant':
            turn_count += 1
            if turn_count >= MAX_TURNS:
                cutoff = i
                break

    trimmed = conversation[cutoff:]
    if len(trimmed) < len(conversation):
        dropped = len(conversation) - len(trimmed)
        print(f'[context] Trimmed {dropped} old message(s) to stay within {MAX_TURNS}-turn window.')

    return list(system_messages) + trimmed


async def _stream_response(
    client,
    model: str,
    system_messages: list,
    conversation: list,
    registry: ToolRegistry,
):
    tp = ToolProcessor(
        registry,
        soul_writer=_write_soul_section,
        soul_reader=_get_soul_section,
        soul_remover=_remove_soul_section,
    )

    messages = _trim_messages(system_messages, conversation)

    response = await client.chat(
        model=model,
        messages=messages,
        think=True,
        stream=True,
        options={'num_ctx': NUM_CTX},
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
    soul_content = _load_soul() or '(soul.md not found)'
    memory_content = _load_memory() or '(memory.json is empty)'

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
        options={'num_ctx': NUM_CTX},
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

    soul_start = full.find('# ')
    if soul_start != -1:
        updated_soul = full[soul_start:].strip()
        with open(SOUL_FILE, 'w', encoding='utf-8') as f:
            f.write(updated_soul + '\n')
        print('\n[soul_update] soul.md updated.')
    else:
        print('\n[soul_update] Could not extract updated soul.md -- no changes written.')


def _print_help(registry: ToolRegistry) -> None:
    print('\n-- REPL Commands ------------------------------------------')
    print('  /?                  show REPL commands and tools')
    print('  /tools              list all registered tools')
    print('  /soul               print soul.md')
    print('  /soul_reset         restore soul.md to default content')
    print('  /soul_update        review memory and update User Profile in soul.md')
    print('  /save_session       save the current session to JSON')
    print('  /load_session <f>   load a saved session from JSON')
    print('  /model <name>       switch the active Ollama model')
    print('  /history            show a summary of the current conversation history')
    print('  /clear              clear conversation history')
    print('  /quit               exit the agent')
    print('\n-- Registered Tools ----------------------------------------')
    for name, meta in registry.all().items():
        danger = '  [DANGEROUS]' if meta.dangerous else ''
        args = ', '.join(meta.arg_names) if meta.arg_names else 'no args'
        print(f'  {name}({args}) -- {meta.description}{danger}')
    print('\n-- Soul Tools (intercepted, not in registry) ---------------')
    print('  propose_soul_edit(section, proposed_content)')
    print('  propose_soul_remove(section)')
    print(f'\n-- Context Settings ----------------------------------------')
    print(f'  NUM_CTX   = {NUM_CTX} tokens')
    print(f'  MAX_TURNS = {MAX_TURNS} turns')
    print()


async def main() -> None:
    registry = ToolRegistry()
    current_model = 'gemma4'

    system_messages = _build_system_messages(registry)
    conversation: list = []

    print('Agent ready.')
    print('Type /? for help.')

    while True:
        user_message = input('\nYou: ').strip()

        if not user_message:
            continue

        if user_message == '/quit':
            break

        if user_message == '/?':
            _print_help(registry)
            continue

        if user_message == '/clear':
            system_messages = _build_system_messages(registry)
            conversation = []
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

        if user_message == '/history':
            if not conversation:
                print('[history] No conversation history yet.')
            else:
                print(f'\n-- Conversation History ({len(conversation)} messages) ----------')
                for i, m in enumerate(conversation):
                    preview = m['content'].replace('\n', ' ')[:80]
                    print(f'  [{i:02d}] {m["role"]:10s}  {preview}')
                print()
            continue

        if user_message == '/soul':
            soul = _load_soul() or SOUL_DEFAULT
            print('\n-- soul.md ----------------------------------------------------\n')
            print(soul)
            print()
            continue

        if user_message == '/soul_reset':
            answer = input('[soul_reset] Restore soul.md to default content? [y/N]: ')
            if answer.strip().lower() in ('y', 'yes'):
                with open(SOUL_FILE, 'w', encoding='utf-8') as f:
                    f.write(SOUL_DEFAULT)
                print('[soul_reset] soul.md restored to default.')
            else:
                print('[soul_reset] Cancelled.')
            continue

        if user_message == '/save_session':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'session_{timestamp}.json'
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)
            print(f'[save_session] Saved {len(conversation)} messages to {filename}')
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
                conversation = loaded
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

        system_messages = _build_system_messages(registry, user_message)

        conversation.append({'role': 'user', 'content': user_message})

        full_response, tool_results = await _stream_response(
            _client, current_model, system_messages, conversation, registry
        )
        conversation.append({'role': 'assistant', 'content': full_response})

        for tr in tool_results:
            conversation.append({
                'role': 'assistant',
                'content': f"I ran {tr['tool']} and got:\n{tr['result']}"
            })

        if tool_results:
            followup_response, _ = await _stream_response(
                _client, current_model, system_messages, conversation, registry
            )
            conversation.append({'role': 'assistant', 'content': followup_response})

    print('\n[Session complete]')


if __name__ == '__main__':
    asyncio.run(main())
