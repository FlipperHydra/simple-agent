import ast
import ipaddress
import json
import os
import shutil
import socket
import urllib.request
import zipfile
from datetime import datetime
from urllib.parse import urlparse
from ddgs import DDGS

from config import MEMORY_FILE, OUTPUT_FILE, atomic_write, atomic_write_json

try:
    import tiktoken
    _ENC = tiktoken.get_encoding('cl100k_base')
except Exception:
    _ENC = None

_CHUNK_TOKENS = 1024

_SUMMARIZE_CHUNK_PROMPT = (
    "Accurately summarize this chunk. Update the anchor notes list with any new "
    "function names, variable names, data structures, key decisions, or facts needed "
    "to reconstruct intent. Anchor notes are lossless checkpoints — never drop a "
    "previous note, only add. Output format:\n"
    "CHUNK SUMMARY: <summary>\n"
    "ANCHOR NOTES:\n<updated full list>"
)

_SUMMARIZE_FINAL_PROMPT = (
    "Using the chunk summaries and anchor notes in this conversation, produce a single "
    "tight technical summary of the full file followed by the complete anchor notes list. "
    "No paraphrasing of technical terms. No filler. Accuracy over readability.\n"
    "Output format:\n"
    "FINAL SUMMARY: <summary>\n"
    "ANCHOR NOTES:\n<complete list>"
)


def _load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {'log': [], 'facts': {}}


def _save_memory(data: dict) -> str | None:
    """Persist memory atomically. Returns an error string on failure instead
    of raising, so a failed write can never crash the current turn."""
    try:
        atomic_write_json(MEMORY_FILE, data)
        return None
    except OSError as e:
        return f'[memory] save failed: {e}'


def _tokenize(text: str) -> list:
    if _ENC:
        return _ENC.encode(text)
    return list(text.encode('utf-8'))


def _decode_tokens(tokens: list) -> str:
    if _ENC:
        return _ENC.decode(tokens)
    return bytes(tokens).decode('utf-8', errors='replace')


def _chunk_text(text: str, chunk_size: int = _CHUNK_TOKENS) -> list[str]:
    tokens = _tokenize(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunks.append(_decode_tokens(tokens[i:i + chunk_size]))
    return chunks


def _is_blocked_host(hostname: str) -> bool:
    """Return True if hostname resolves to any non-public address.

    Resolves the host and inspects every returned IP; if resolution fails or
    any address is private/loopback/link-local/reserved/multicast/unspecified
    the request is blocked. This defends against SSRF to internal services
    (e.g. cloud metadata at 169.254.169.254) and local-file exfiltration.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True
    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            return True
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            return True
    return False


# In-memory store for active summarization sessions keyed by filename
_summarize_sessions: dict[str, dict] = {}


class Tools:
    """
    All agent tool implementations as static methods.
    Import and call directly for testing:
        from tools import Tools
        Tools.read_file('output.txt')
    """

    @staticmethod
    def write_tool(text: str) -> None:
        formatted = (
            text
            .replace('\\n', '\n')
            .replace('\\t', '\t')
            .replace("\\'", "'")
        )
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            f.write(formatted + '\n')
        print(f'\n[write_tool] Wrote -> {formatted}')

    @staticmethod
    def save_tool(filename: str, content: str) -> str:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(content + '\n')
        print(f'\n[save_tool] Saved -> {filename}')
        return f'saved to {filename}'

    @staticmethod
    def overwrite_file(filename: str, content: str) -> str:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'\n[overwrite_file] Overwrote -> {filename}')
            return f'[overwrite_file] Written: {filename}'
        except Exception as e:
            return f'[overwrite_file] Error: {e}'

    @staticmethod
    def get_datetime() -> str:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'\n[get_datetime] {now}')
        return now

    @staticmethod
    def read_file(filename: str) -> str:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f'[read_file] File not found: {filename}'

    @staticmethod
    def list_files(directory: str = '.') -> str:
        try:
            entries = os.listdir(directory)
            return '\n'.join(entries)
        except FileNotFoundError:
            return f'[list_files] Directory not found: {directory}'

    @staticmethod
    def fetch_url(url: str) -> str:
        # SSRF / local-file-read guard: only http(s), and never resolve to a
        # private, loopback, or link-local address (blocks file://, ftp://,
        # 127.0.0.0/8, 10/8, 172.16/12, 192.168/16, 169.254/16, ::1, etc.).
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return '[fetch_url] Blocked: only http and https URLs are allowed.'
        host = parsed.hostname
        if not host:
            return '[fetch_url] Blocked: URL has no host.'
        if _is_blocked_host(host):
            return (
                '[fetch_url] Blocked: refusing to fetch a private, loopback, '
                'link-local, or otherwise non-public address.'
            )
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode('utf-8', errors='replace')[:4000]
        except Exception as e:
            return f'[fetch_url] Error: {e}'

    @staticmethod
    def search_web(query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return '[search_web] No results found.'
            lines = [
                f"{r['title']}: {r['body']} ({r['href']})"
                for r in results
            ]
            return '\n'.join(lines)
        except Exception as e:
            return f'[search_web] Error: {e}'

    @staticmethod
    def delete_file(filename: str) -> str:
        try:
            os.remove(filename)
            return f'[delete_file] Deleted: {filename}'
        except FileNotFoundError:
            return f'[delete_file] File not found: {filename}'
        except Exception as e:
            return f'[delete_file] Error: {e}'

    @staticmethod
    def make_directory(path: str) -> str:
        try:
            os.makedirs(path, exist_ok=True)
            return f'[make_directory] Created: {path}'
        except Exception as e:
            return f'[make_directory] Error: {e}'

    @staticmethod
    def copy_file(src: str, dest: str) -> str:
        try:
            shutil.copy2(src, dest)
            print(f'\n[copy_file] {src} -> {dest}')
            return f'[copy_file] Copied {src} to {dest}'
        except FileNotFoundError:
            return f'[copy_file] File not found: {src}'
        except Exception as e:
            return f'[copy_file] Error: {e}'

    @staticmethod
    def move_file(src: str, dest: str) -> str:
        try:
            shutil.move(src, dest)
            print(f'\n[move_file] {src} -> {dest}')
            return f'[move_file] Moved {src} to {dest}'
        except FileNotFoundError:
            return f'[move_file] File not found: {src}'
        except Exception as e:
            return f'[move_file] Error: {e}'

    @staticmethod
    def append_memory(note: str) -> str:
        data = _load_memory()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        entry = {'timestamp': timestamp, 'note': note}
        data['log'].append(entry)
        err = _save_memory(data)
        if err:
            return err
        print(f'\n[append_memory] Stored -> {entry}')
        return f'[append_memory] Stored note at {timestamp}'

    @staticmethod
    def recall_memory() -> str:
        data = _load_memory()
        log = data.get('log', [])
        facts = data.get('facts', {})
        if not log and not facts:
            return '[recall_memory] Memory is empty.'
        lines = []
        if facts:
            lines.append('--- Facts ---')
            for k, v in facts.items():
                lines.append(f'  {k}: {v}')
        if log:
            lines.append('--- Log ---')
            for entry in log:
                lines.append(f"  [{entry['timestamp']}] {entry['note']}")
        return '\n'.join(lines)

    @staticmethod
    def clear_memory() -> str:
        err = _save_memory({'log': [], 'facts': {}})
        if err:
            return err
        print('\n[clear_memory] memory.json cleared.')
        return '[clear_memory] Memory cleared.'

    @staticmethod
    def write_json(filename: str, key: str, value: str) -> str:
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {}
            data[key] = value
            atomic_write_json(filename, data)
            print(f'\n[write_json] {filename}[{key}] = {value}')
            return f'[write_json] Set {key} in {filename}'
        except Exception as e:
            return f'[write_json] Error: {e}'

    @staticmethod
    def read_json(filename: str, key: str = '') -> str:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if key:
                if key in data:
                    return str(data[key])
                return f'[read_json] Key not found: {key}'
            return json.dumps(data, indent=2, ensure_ascii=False)
        except FileNotFoundError:
            return f'[read_json] File not found: {filename}'
        except Exception as e:
            return f'[read_json] Error: {e}'

    # Exponents above this are refused: a**b with a large b can allocate a
    # gigantic integer (e.g. 9**9**9) and hang the process / exhaust memory.
    # 1000 comfortably covers legitimate math while blocking the DoS case.
    _MAX_POW_EXPONENT = 1000

    @staticmethod
    def eval_math(expression: str) -> str:
        _ALLOWED_NODES = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
            ast.Mod, ast.FloorDiv, ast.USub, ast.UAdd,
        )
        try:
            tree = ast.parse(expression.strip(), mode='eval')
            for node in ast.walk(tree):
                if not isinstance(node, _ALLOWED_NODES):
                    return f'[eval_math] Disallowed expression: {type(node).__name__}'
                # Guard exponentiation before evaluating so we never actually
                # compute an unsafe power. A constant exponent must be within
                # the bound; a *computed* exponent (e.g. 9**9**9, where the
                # exponent is itself an expression) is refused outright because
                # its magnitude cannot be cheaply bounded ahead of time.
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                    exp = node.right
                    if isinstance(exp, ast.Constant) and isinstance(exp.value, (int, float)):
                        if abs(exp.value) > Tools._MAX_POW_EXPONENT:
                            return (
                                '[eval_math] Exponent too large to compute safely '
                                f'(limit is {Tools._MAX_POW_EXPONENT}); please use a '
                                'smaller exponent or approximate with floating point.'
                            )
                    else:
                        return (
                            '[eval_math] Exponent too large or complex to compute '
                            'safely; please use a small constant exponent or '
                            'approximate with floating point.'
                        )
            result = eval(compile(tree, '<string>', 'eval'))  # noqa: S307
            print(f'\n[eval_math] {expression} = {result}')
            return str(result)
        except Exception as e:
            return f'[eval_math] Error: {e}'

    @staticmethod
    def summarize_file(filename: str) -> str:
        """
        Init step: reads the file, calculates chunks, stores session,
        and instructs the agent to call summarize_file_chunk for each index.
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return f'[summarize_file] File not found: {filename}'
        except Exception as e:
            return f'[summarize_file] Error reading file: {e}'

        chunks = _chunk_text(content)
        total = len(chunks)
        token_count = len(_tokenize(content))

        _summarize_sessions[filename] = {
            'chunks': chunks,
            'total': total,
            'mini_summaries': [],
            'anchor_notes': '',
        }

        print(f'\n[summarize_file] {filename}: {token_count} tokens, {total} chunk(s)')
        return (
            f'[summarize_file] Session ready for "{filename}": {total} chunk(s), '
            f'{token_count} total tokens.\n'
            f'Call summarize_file_chunk("{filename}", chunk_index) for each index 0 to {total - 1} in order.\n'
            f'When all chunks are processed, call summarize_file_finalize("{filename}") to synthesize the final summary.\n'
            f'Chunk prompt to use for each call:\n{_SUMMARIZE_CHUNK_PROMPT}'
        )

    @staticmethod
    def summarize_file_chunk(filename: str, chunk_index: int, chunk_summary: str) -> str:
        """
        Per-chunk step: stores the agent-provided summary and updated anchor notes
        for this chunk index. The agent calls this after processing each chunk.
        chunk_summary should contain the agent output for this chunk including
        the updated anchor notes section.
        """
        session = _summarize_sessions.get(filename)
        if not session:
            return f'[summarize_file_chunk] No active session for "{filename}". Call summarize_file first.'

        total = session['total']
        if chunk_index < 0 or chunk_index >= total:
            return f'[summarize_file_chunk] chunk_index {chunk_index} out of range (0-{total - 1}).'

        lower = chunk_summary.lower()
        if 'anchor notes' in lower:
            split = lower.find('anchor notes')
            session['anchor_notes'] = chunk_summary[split:].strip()

        session['mini_summaries'].append(f'[Chunk {chunk_index}]\n{chunk_summary}')

        remaining = total - len(session['mini_summaries'])
        print(f'\n[summarize_file_chunk] Stored chunk {chunk_index}/{total - 1} for "{filename}"')

        if remaining > 0:
            next_chunk = session['chunks'][chunk_index + 1] if chunk_index + 1 < total else ''
            return (
                f'[summarize_file_chunk] Chunk {chunk_index} stored. {remaining} chunk(s) remaining.\n'
                f'Next chunk ({chunk_index + 1} of {total - 1}):\n---\n{next_chunk}\n---\n'
                f'Current anchor notes:\n{session["anchor_notes"]}'
            )
        else:
            return (
                f'[summarize_file_chunk] All {total} chunks processed. '
                f'Call summarize_file_finalize("{filename}") to synthesize.'
            )

    @staticmethod
    def summarize_file_finalize(filename: str) -> str:
        """
        Final step: returns all mini-summaries and accumulated anchor notes
        with the synthesis prompt, instructing the agent to produce the final output.
        """
        session = _summarize_sessions.get(filename)
        if not session:
            return f'[summarize_file_finalize] No active session for "{filename}". Call summarize_file first.'

        mini_summaries = session['mini_summaries']
        anchor_notes = session['anchor_notes']

        del _summarize_sessions[filename]
        print(f'\n[summarize_file_finalize] Synthesizing {len(mini_summaries)} chunk(s) for "{filename}"')

        return (
            f'{_SUMMARIZE_FINAL_PROMPT}\n\n'
            + '\n\n'.join(mini_summaries)
            + f'\n\nFull anchor notes:\n{anchor_notes}'
        )

    @staticmethod
    def zip_files(filenames_csv: str, output_zip: str) -> str:
        try:
            filenames = [f.strip() for f in filenames_csv.split(',') if f.strip()]
            with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
                for fname in filenames:
                    if os.path.exists(fname):
                        zf.write(fname)
                    else:
                        print(f'[zip_files] Skipping missing file: {fname}')
            print(f'\n[zip_files] Created {output_zip} with {len(filenames)} file(s)')
            return f'[zip_files] Archive created: {output_zip}'
        except Exception as e:
            return f'[zip_files] Error: {e}'

    @staticmethod
    def propose_soul_edit(section: str, proposed_content: str) -> str:
        return f'[propose_soul_edit] Proposed update for section: {section}'
