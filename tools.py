import os
import urllib.request
from datetime import datetime
from ddgs import DDGS


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
    def append_memory(note: str) -> str:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        entry = f'[{timestamp}] {note}'
        with open('memory.md', 'a', encoding='utf-8') as f:
            f.write(entry + '\n')
        print(f'\n[append_memory] Stored -> {entry}')
        return f'[append_memory] Stored note at {timestamp}'

    @staticmethod
    def recall_memory() -> str:
        try:
            with open('memory.md', 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if not content:
                return '[recall_memory] memory.md is empty.'
            return content
        except FileNotFoundError:
            return '[recall_memory] No memory file found yet.'

    @staticmethod
    def propose_soul_edit(section: str, proposed_content: str) -> str:
        return f'[propose_soul_edit] Proposed update for section: {section}'
