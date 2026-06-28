import os
import io
import sys
import json
import datetime
import urllib.request
import urllib.parse


class Tools:
    """
    All agent tool implementations as static methods.
    Import and call directly for testing:
        from tools import Tools
        Tools.read_file('output.txt')
    """

    # ------------------------------------------------------------------
    # Existing tools
    # ------------------------------------------------------------------

    @staticmethod
    def write_tool(text: str) -> None:
        formatted = (
            text
            .replace('\\n', '\n')
            .replace('\\t', '\t')
            .replace("\\\'" , "'")
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
                headers={'User-Agent': 'Mozilla/5.0 (simple-agent research tool)'}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read().decode('utf-8', errors='replace')[:8000]
        except Exception as e:
            return f'[fetch_url] Error: {e}'

    @staticmethod
    def search_web(query: str) -> str:
        try:
            encoded = urllib.parse.quote(query)
            url = f'https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1'
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            abstract = data.get('AbstractText', '')
            source_url = data.get('AbstractURL', '')
            results = []
            for r in data.get('RelatedTopics', [])[:8]:
                text = r.get('Text', '')
                first_url = r.get('FirstURL', '')
                if text:
                    results.append(f'{text}\n  URL: {first_url}' if first_url else text)
            combined = ''
            if abstract:
                combined += f'{abstract}\n  Source: {source_url}\n\n'
            combined += '\n\n'.join(results)
            return combined.strip() or '[search_web] No results found.'
        except Exception as e:
            return f'[search_web] Error: {e}'

    # ------------------------------------------------------------------
    # New tools from plan
    # ------------------------------------------------------------------

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
    def get_datetime() -> str:
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def summarize_file(filename: str, max_chars: str = '2000') -> str:
        try:
            limit = int(max_chars)
        except ValueError:
            limit = 2000
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            if len(content) <= limit:
                return content
            return content[:limit] + f'\n... [{len(content) - limit} more chars truncated]'
        except FileNotFoundError:
            return f'[summarize_file] File not found: {filename}'

    @staticmethod
    def append_memory(note: str) -> str:
        memory_file = 'memory.md'
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        entry = f'[{timestamp}] {note}\n'
        with open(memory_file, 'a', encoding='utf-8') as f:
            f.write(entry)
        return f'[append_memory] Stored: {note[:80]}'

    @staticmethod
    def recall_memory() -> str:
        memory_file = 'memory.md'
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return content if content else '[recall_memory] Memory is empty.'
        except FileNotFoundError:
            return '[recall_memory] No memory file found.'

    # ------------------------------------------------------------------
    # Research-specific tools
    # ------------------------------------------------------------------

    @staticmethod
    def search_and_fetch(query: str) -> str:
        """Search the web, then fetch the top result URL for full content."""
        try:
            encoded = urllib.parse.quote(query)
            url = f'https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1'
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())

            abstract = data.get('AbstractText', '')
            source_url = data.get('AbstractURL', '')
            related = data.get('RelatedTopics', [])

            # Try to get the top URL to fetch full content
            top_url = source_url
            if not top_url and related:
                top_url = related[0].get('FirstURL', '')

            summary = ''
            if abstract:
                summary += f'Abstract: {abstract}\nSource: {source_url}\n\n'
            for r in related[:5]:
                text = r.get('Text', '')
                furl = r.get('FirstURL', '')
                if text:
                    summary += f'{text}\n  URL: {furl}\n'

            if top_url:
                fetched = Tools.fetch_url(top_url)
                summary += f'\n-- Full content from {top_url} (first 4000 chars) --\n{fetched[:4000]}'

            return summary.strip() or '[search_and_fetch] No results found.'
        except Exception as e:
            return f'[search_and_fetch] Error: {e}'

    @staticmethod
    def multi_search(queries_json: str) -> str:
        """Run up to 5 search_web queries from a JSON array and return combined results.

        queries_json: JSON array of query strings, e.g. ["query one", "query two"]
        Results are labeled by query so the agent can distinguish them.
        """
        try:
            queries = json.loads(queries_json)
        except json.JSONDecodeError as e:
            return f'[multi_search] Invalid JSON: {e}\nPass a JSON array of strings.'

        if not isinstance(queries, list):
            return '[multi_search] Expected a JSON array of query strings.'

        queries = [str(q) for q in queries[:5]]
        parts = []
        for i, q in enumerate(queries, 1):
            result = Tools.search_web(q)
            parts.append(f'-- Query {i}: {q} --\n{result}')

        return '\n\n'.join(parts)
