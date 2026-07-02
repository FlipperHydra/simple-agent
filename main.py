from __future__ import annotations
import asyncio
import json
import os
import re
import shutil
from datetime import datetime
import ollama
from config import (
    SOUL_FILE,
    MEMORY_FILE,
    CONVERSATION_FILE,
    DATA_DIR,
    data_path,
    atomic_write,
    atomic_write_json,
)
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
    compaction_prompt,
)

try:
    import tiktoken
    _ENC = tiktoken.get_encoding('cl100k_base')
except Exception:
    _ENC = None

_client = ollama.AsyncClient()

# Model is configurable; gemma4 (https://ollama.com/library/gemma4) is a real,
# valid Ollama model and remains the default. Override with AGENT_MODEL.
DEFAULT_MODEL = os.environ.get('AGENT_MODEL', 'gemma4')

# Context window passed to Ollama on every chat call.
# 16384 tokens gives comfortable room for system prompts (~4k) plus long conversations.
# Raise to 32768 if your hardware supports it; lower to 8192 to save VRAM.
NUM_CTX = int(os.environ.get('AGENT_NUM_CTX', '16384'))

# --- Context compaction (replaces the old MAX_TURNS hard-truncation) --------
# When the conversation exceeds this many tokens we offer to COMPACT (summarize)
# the oldest half rather than silently dropping messages. Configurable.
COMPACT_THRESHOLD_TOKENS = int(os.environ.get('AGENT_COMPACT_THRESHOLD', '2000'))
# After the user accepts or declines a compaction prompt, we do not nag every
# turn: we only re-prompt once the conversation has grown by more than this
# many tokens since that last decision.
COMPACT_REPROMPT_DELTA = int(os.environ.get('AGENT_COMPACT_REPROMPT_DELTA', '250'))

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
E. Do not narrate routine tool mechanics in your replies (e.g. do not
   explain that you called a tool unless the user directly asks why).
   Act on tools silently and speak naturally about outcomes, not process.

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


def _backup_and_write_soul(content: str) -> bool:
    """Write soul.md safely: back up the current file to soul.md.bak first,
    then write atomically (temp file + os.replace). Returns True on success.

    soul.md is the agent's persisted identity/profile; a truncated or garbage
    write here silently corrupts it, so every write goes through this path.
    """
    try:
        if os.path.exists(SOUL_FILE):
            shutil.copy2(SOUL_FILE, SOUL_FILE + '.bak')
    except OSError as e:
        print(f'[soul] warning: could not back up soul.md: {e}')
    try:
        atomic_write(SOUL_FILE, content)
        return True
    except OSError as e:
        print(f'[soul] error: could not write soul.md: {e}')
        return False


def _seed_default_files() -> None:
    """Seed the data directory with the repo's shipped soul.md on first run so
    behaviour matches the pre-data-dir version. Only runs when no soul.md
    exists yet in the data directory."""
    repo_soul = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soul.md')
    if (not os.path.exists(SOUL_FILE) and os.path.exists(repo_soul)
            and os.path.abspath(repo_soul) != os.path.abspath(SOUL_FILE)):
        try:
            shutil.copy2(repo_soul, SOUL_FILE)
        except OSError:
            pass


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

    if _backup_and_write_soul(updated.strip() + '\n'):
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

    if _backup_and_write_soul(updated.strip() + '\n'):
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


# --------------------------------------------------------------------------
# Conversation persistence
# --------------------------------------------------------------------------
def _load_conversation() -> list:
    """Load the persisted conversation from disk (empty list if none/invalid)."""
    if os.path.exists(CONVERSATION_FILE):
        try:
            with open(CONVERSATION_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_conversation(conversation: list) -> None:
    """Autosave conversation atomically. Warns (never crashes) on failure."""
    try:
        atomic_write_json(CONVERSATION_FILE, conversation)
    except OSError as e:
        print(f'[warn] could not save conversation: {e}')


# --------------------------------------------------------------------------
# Token counting + context compaction (replaces MAX_TURNS truncation)
# --------------------------------------------------------------------------
def _count_tokens(messages) -> int:
    """Token count of a message list (or raw string) via tiktoken.

    Falls back to a ~4-chars-per-token heuristic when tiktoken is unavailable.
    """
    if isinstance(messages, str):
        text = messages
    else:
        text = '\n'.join(str(m.get('content', '')) for m in messages)
    if _ENC:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


def _render_conversation(messages: list) -> str:
    return '\n\n'.join(f"{m['role'].upper()}: {m.get('content', '')}" for m in messages)


def _split_for_compaction(conversation: list) -> int | None:
    """Choose an index splitting conversation into (older, recent).

    Split points are only ever at user-message boundaries so a user/assistant
    pair is never torn apart. We aim to move roughly the oldest half (by tokens)
    into the older segment while always keeping at least the most recent
    complete turn verbatim. Returns None when there is nothing safe to compact
    (fewer than two turns).
    """
    boundaries = [i for i, m in enumerate(conversation) if m['role'] == 'user']
    if len(boundaries) < 2:
        return None
    total = _count_tokens(conversation)
    half = total / 2
    # Default: keep only the final turn verbatim, compact everything before it.
    split = boundaries[-1]
    for b in boundaries[1:]:  # never split before the first turn
        if _count_tokens(conversation[:b]) >= half:
            split = b
            break
    return split


async def _summarize_segment(client, model: str, segment: list) -> str | None:
    """Ask the model to compact an older conversation segment. Returns the
    summary text, or None if the call fails (caller keeps the raw segment)."""
    prompt = compaction_prompt(_render_conversation(segment))
    try:
        resp = await client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            options={'num_ctx': NUM_CTX},
        )
    except Exception as e:
        print(f'[compact] summarization failed, keeping messages intact: {e}')
        return None
    return (resp.message.content or '').strip()


async def _compact_once(client, model: str, conversation: list) -> list:
    """Perform a single compaction pass. Older segment is replaced by one
    synthetic system message containing its summary. On failure or when there
    is nothing to compact, returns the conversation unchanged."""
    split = _split_for_compaction(conversation)
    if not split:
        return conversation
    older, recent = conversation[:split], conversation[split:]
    summary = await _summarize_segment(client, model, older)
    if not summary:
        return conversation
    compacted = {
        'role': 'system',
        'content': f'[Compacted summary of earlier conversation]: {summary}',
    }
    print(f'[compact] Summarized {len(older)} older message(s) into 1 summary.')
    return [compacted] + recent


async def _maybe_compact(
    client,
    model: str,
    conversation: list,
    tokens_at_last_prompt: int,
) -> tuple[list, int]:
    """Offer/perform compaction according to the token budget policy.

    - Hard cap: if tokens exceed the model's context window (NUM_CTX), compact
      automatically without asking (repeatedly, until safe).
    - Soft threshold: above COMPACT_THRESHOLD_TOKENS, ask the user Y/N, but only
      re-ask once growth since the last decision exceeds COMPACT_REPROMPT_DELTA.
    Returns the (possibly compacted) conversation and the updated
    tokens-at-last-prompt marker.
    """
    total = _count_tokens(conversation)

    # Hard cap -- force compaction, no prompt.
    if total > NUM_CTX:
        print(
            f'[auto-compact] Context reached the model\'s hard limit '
            f'(NUM_CTX={NUM_CTX}); compacting automatically to prevent corruption.'
        )
        while _count_tokens(conversation) > NUM_CTX:
            new_conv = await _compact_once(client, model, conversation)
            if new_conv is conversation:  # cannot reduce further
                break
            conversation = new_conv
        _save_conversation(conversation)
        return conversation, _count_tokens(conversation)

    # Soft threshold -- prompt the user, respecting the re-prompt delta.
    if total > COMPACT_THRESHOLD_TOKENS and (total - tokens_at_last_prompt) > COMPACT_REPROMPT_DELTA:
        answer = await asyncio.to_thread(
            input,
            'Conversation context exceeding 2000 tokens. In order to preserve '
            'current context, compact older messages? Y/N ',
        )
        if answer.strip().lower() in ('y', 'yes'):
            conversation = await _compact_once(client, model, conversation)
            # Keep compacting if still over the soft threshold.
            while _count_tokens(conversation) > COMPACT_THRESHOLD_TOKENS:
                new_conv = await _compact_once(client, model, conversation)
                if new_conv is conversation:
                    break
                conversation = new_conv
            _save_conversation(conversation)
        # Whether accepted or declined, record the marker so we don't re-prompt
        # until growth exceeds the delta again.
        return conversation, _count_tokens(conversation)

    return conversation, tokens_at_last_prompt


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

    # Compaction keeps `conversation` within budget, so we send it in full;
    # system messages are always pinned at the front.
    messages = list(system_messages) + conversation

    # A single failed chat call must never kill the session (and thus the whole
    # conversation/context). On error we print and return an empty result so the
    # REPL loop can continue.
    try:
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
    except Exception as e:
        print(f'\n[error] chat failed: {e}')
        return '', []

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
    try:
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
    except Exception as e:
        print(f'\n[error] chat failed: {e}')
        print('[soul_update] Aborted -- no changes written.')
        return

    print()

    # Extract the new soul.md ONLY from between the strict sentinels the prompt
    # requires. The previous implementation grabbed everything after the first
    # occurrence of the substring '# ' in the model's raw output. That output
    # includes the model's unfiltered reasoning/commentary, so any stray '# '
    # earlier in that text (a markdown-style heading in prose, a '#' code
    # comment being discussed, etc.) caused the WRONG slice to be written to
    # soul.md -- silently overwriting the persisted identity/profile with
    # garbage, with no backup and no validation. We now require BOTH
    # ===SOUL START===/===SOUL END=== sentinels, write only what is between
    # them, back up the previous soul.md, and write atomically.
    start_tag, end_tag = '===SOUL START===', '===SOUL END==='
    start = full.find(start_tag)
    end = full.find(end_tag, start + len(start_tag)) if start != -1 else -1
    if start == -1 or end == -1:
        print('\n[soul_update] Sentinels not found in output -- no changes written.')
        return

    updated_soul = full[start + len(start_tag):end].strip()
    if not updated_soul.startswith('#'):
        print('\n[soul_update] Extracted content is not a valid soul document -- no changes written.')
        return

    if _backup_and_write_soul(updated_soul + '\n'):
        print('\n[soul_update] soul.md updated (previous version saved to soul.md.bak).')


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
    print(f'  NUM_CTX                  = {NUM_CTX} tokens (model hard limit)')
    print(f'  COMPACT_THRESHOLD_TOKENS = {COMPACT_THRESHOLD_TOKENS} tokens (offer to compact)')
    print(f'  COMPACT_REPROMPT_DELTA   = {COMPACT_REPROMPT_DELTA} tokens')
    print(f'  DATA_DIR                 = {DATA_DIR}')
    print()


async def _validate_model(client, model: str) -> bool:
    """Fail fast if the configured model is not pulled locally, instead of
    crashing deep inside the first chat call."""
    try:
        resp = await client.list()
    except Exception as e:
        print(f'[startup] Could not reach Ollama to list models: {e}')
        print('[startup] Is the Ollama server running? Try:  ollama serve')
        return False

    names = []
    for m in (getattr(resp, 'models', None) or []):
        name = getattr(m, 'model', None)
        if name is None and isinstance(m, dict):
            name = m.get('model') or m.get('name')
        if name:
            names.append(name)

    base = model.split(':')[0]
    if any(n == model or n.split(':')[0] == base for n in names):
        return True

    print(f"[startup] Model '{model}' is not available locally.")
    print(f'[startup] Pull it first with:  ollama pull {model}')
    print(f"[startup] Models found: {', '.join(names) if names else '(none)'}")
    return False


async def main() -> None:
    registry = ToolRegistry()
    current_model = DEFAULT_MODEL

    # Seed the data dir and validate the model before entering the REPL.
    _seed_default_files()
    if not await _validate_model(_client, current_model):
        return

    system_messages = _build_system_messages(registry)
    # Autoload persisted conversation so context survives restarts.
    conversation: list = _load_conversation()
    tokens_at_last_prompt = _count_tokens(conversation)

    print('Agent ready.')
    if conversation:
        print(f'[context] Restored {len(conversation)} message(s) from {CONVERSATION_FILE}')
    print('Type /? for help.')

    while True:
        try:
            user_message = input('\nYou: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

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
            tokens_at_last_prompt = 0
            _save_conversation(conversation)
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
                if _backup_and_write_soul(SOUL_DEFAULT):
                    print('[soul_reset] soul.md restored to default.')
            else:
                print('[soul_reset] Cancelled.')
            continue

        if user_message == '/save_session':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = data_path(f'session_{timestamp}.json')
            try:
                atomic_write_json(filepath, conversation)
                print(f'[save_session] Saved {len(conversation)} messages to {filepath}')
            except OSError as e:
                print(f'[save_session] Error: {e}')
            continue

        if user_message.startswith('/load_session'):
            parts = user_message.split(maxsplit=1)
            if len(parts) < 2:
                print('[load_session] Usage: /load_session <filename>')
                continue
            filepath = parts[1].strip()
            # Accept a bare filename saved in the data dir as well as a path.
            if not os.path.exists(filepath) and os.path.exists(data_path(filepath)):
                filepath = data_path(filepath)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                conversation = loaded
                tokens_at_last_prompt = _count_tokens(conversation)
                _save_conversation(conversation)
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

        # Any unexpected failure in a single turn must not tear down the REPL
        # (which would drop the in-memory conversation). Catch, report, continue.
        try:
            system_messages = _build_system_messages(registry, user_message)

            conversation.append({'role': 'user', 'content': user_message})

            full_response, tool_results = await _stream_response(
                _client, current_model, system_messages, conversation, registry
            )
            conversation.append({'role': 'assistant', 'content': full_response})

            # Tool outputs are external observations, not the model's own words,
            # so they use role 'tool' (previously mislabeled 'assistant').
            for tr in tool_results:
                conversation.append({
                    'role': 'tool',
                    'content': f"[tool result: {tr['tool']}]\n{tr['result']}"
                })

            if tool_results:
                followup_response, _ = await _stream_response(
                    _client, current_model, system_messages, conversation, registry
                )
                conversation.append({'role': 'assistant', 'content': followup_response})

            # Token-budgeted compaction (offer, or auto at the hard cap), then
            # autosave so context survives restarts.
            conversation, tokens_at_last_prompt = await _maybe_compact(
                _client, current_model, conversation, tokens_at_last_prompt
            )
            _save_conversation(conversation)
        except Exception as e:
            print(f'[error] turn failed: {e}')
            _save_conversation(conversation)
            continue

    print('\n[Session complete]')


if __name__ == '__main__':
    asyncio.run(main())
