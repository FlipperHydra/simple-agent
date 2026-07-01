import ast
import json
import os
import shutil
import urllib.request
import zipfile
from datetime import datetime
from ddgs import DDGS

try:
    import tiktoken
    _ENC = tiktoken.get_encoding('cl100k_base')
except Exception:
    _ENC = None

MEMORY_FILE = 'memory.json'
_CHUNK_TOKENS = 200

_SUMMARIZE_SYSTEM = (
    "Accurately summarize this chunk. Update the anchor notes list with any new "
    "function names, variable names, data structures, key decisions, or facts needed "
    "to reconstruct intent. Anchor notes are lossless checkpoints — never drop a "
    "previous note, only add. Output: chunk summary, then full updated anchor notes list."
)

_SUMMARIZE_FINAL = (
    "Using the mini-summaries and anchor notes below, produce a single tight technical "
    "summary of the full file followed by the complete anchor notes list. "
    "No paraphrasing of technical terms. No filler. Accuracy over readability."
)


def _load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {'log': [], 'facts': {}}


def _save_memory(data: dict) -> None:
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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
            .replace("\\'\', "'")
        )
        with open('output.txt', 'a', encoding='utf-8') as f:
            f.write(formatted + '\n')
        print(f'\n[write_tool] Wrote -> {formatted}')

    @staticmethod
    def save_tool(filename: str, content: str) -> None:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(content + '\n')
        print(f'\n[save_tool] Saved -> {filename}')

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
        _save_memory(data)
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
        _save_memory({'log': [], 'facts': {}})
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
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
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
            result = eval(compile(tree, '<string>', 'eval'))  # noqa: S307
            print(f'\n[eval_math] {expression} = {result}')
            return str(result)
        except Exception as e:
            return f'[eval_math] Error: {e}'

    @staticmethod
    def summarize_file(filename: str, llm_caller=None) -> str:
        """
        Reads a file, chunks it by token count, and iteratively summarizes
        with carry-forward anchor notes. llm_caller must be a sync callable
        that takes a prompt string and returns a string response.
        If llm_caller is None, returns the raw chunked content with the
        summarization prompt prepended for the agent to handle inline.
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return f'[summarize_file] File not found: {filename}'
        except Exception as e:
            return f'[summarize_file] Error reading file: {e}'

        chunks = _chunk_text(content)

        if llm_caller is None:
            # Inline mode: return prompt + full content for the agent loop to handle
            return (
                f"{_SUMMARIZE_SYSTEM}\n\n"
                f"[File: {filename} | {len(chunks)} chunk(s) | "
                f"{len(_tokenize(content))} tokens]\n\n"
                + content
            )

        # Iterative mode: process each chunk with carry-forward anchor notes
        anchor_notes = ''
        mini_summaries = []

        for i, chunk in enumerate(chunks):
            prompt = (
                f"{_SUMMARIZE_SYSTEM}\n\n"
                f"Chunk {i + 1} of {len(chunks)}:\n"
                f"{('Anchor notes so far:\n' + anchor_notes + chr(10)) if anchor_notes else ''}"
                f"---\n{chunk}\n---"
            )
            result = llm_caller(prompt)
            mini_summaries.append(result)

            # Extract updated anchor notes from the result
            if 'anchor notes' in result.lower():
                split = result.lower().find('anchor notes')
                anchor_notes = result[split:].strip()

        # Final synthesis pass
        synthesis_prompt = (
            f"{_SUMMARIZE_FINAL}\n\n"
            + '\n\n'.join(f'[Chunk {i+1}]\n{s}' for i, s in enumerate(mini_summaries))
            + f'\n\nFull anchor notes:\n{anchor_notes}'
        )
        return llm_caller(synthesis_prompt)

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
