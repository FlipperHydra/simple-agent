import os
import json
import urllib.request
import urllib.parse


class Tools:
    """
    All agent tool implementations as static methods.
    Import and call directly for testing:
        from tools import Tools
        Tools.read_file("output.txt")
    """

    @staticmethod
    def write_tool(text: str) -> None:
        formatted = (
            text
            .replace('\\n', '\n')
            .replace('\\t', '\t')
            .replace("\\\'" , "'")
        )
        with open("output.txt", "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
        print(f'\n[write_tool] Wrote \u2192 {formatted}')

    @staticmethod
    def save_tool(filename: str, content: str) -> None:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(content + "\n")
        print(f'\n[save_tool] Saved \u2192 {filename}')

    @staticmethod
    def read_file(filename: str) -> str:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return f"[read_file] File not found: {filename}"

    @staticmethod
    def list_files(directory: str = ".") -> str:
        try:
            entries = os.listdir(directory)
            return "\n".join(entries)
        except FileNotFoundError:
            return f"[list_files] Directory not found: {directory}"

    @staticmethod
    def fetch_url(url: str) -> str:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return resp.read().decode("utf-8", errors="replace")[:4000]
        except Exception as e:
            return f"[fetch_url] Error: {e}"

    @staticmethod
    def search_web(query: str) -> str:
        try:
            encoded = urllib.parse.quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1"
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read())
            abstract = data.get("AbstractText", "")
            results = [r.get("Text", "") for r in data.get("RelatedTopics", [])[:5]]
            combined = abstract + "\n" + "\n".join(results)
            return combined.strip() or "[search_web] No results found."
        except Exception as e:
            return f"[search_web] Error: {e}"
