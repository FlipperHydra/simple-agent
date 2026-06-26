from typing import Any, Callable, Dict, List, NamedTuple
from tools import Tools


# Tool metadata


class ToolMeta(NamedTuple):
    func: Callable[..., Any]
    arg_names: List[str]   # ordered list of argument labels shown in the prompt
    description: str       # one-line description shown in the prompt


# Tool Registry


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolMeta] = {}
        self._register_defaults()

    # MODIFY THIS TO ADD TOOLS
    def _register_defaults(self) -> None:
        self.register_tool(
            "write_tool",
            Tools.write_tool,
            arg_names=["text"],
            description="writes the given text to output.txt",
        )
        self.register_tool(
            "save_tool",
            Tools.save_tool,
            arg_names=["filename", "content"],
            description="saves content to a named file",
        )
        self.register_tool(
            "read_file",
            Tools.read_file,
            arg_names=["filename"],
            description="reads and returns the contents of a file",
        )
        self.register_tool(
            "list_files",
            Tools.list_files,
            arg_names=["directory"],
            description="lists all files in a directory",
        )
        self.register_tool(
            "fetch_url",
            Tools.fetch_url,
            arg_names=["url"],
            description="fetches and returns the text content of a URL (max 4000 chars)",
        )
        self.register_tool(
            "search_web",
            Tools.search_web,
            arg_names=["query"],
            description="searches DuckDuckGo and returns a summary of results",
        )

    def register_tool(
        self,
        name: str,
        func: Callable[..., Any],
        *,
        arg_names: List[str] | None = None,
        description: str = "",
    ) -> None:
        self._tools[name] = ToolMeta(
            func=func,
            arg_names=arg_names or ["arg"],
            description=description,
        )

    def get(self, name: str) -> Callable[..., Any] | None:
        meta = self._tools.get(name)
        return meta.func if meta else None

    def meta(self, name: str) -> ToolMeta | None:
        return self._tools.get(name)

    def all(self) -> Dict[str, ToolMeta]:
        return dict(self._tools)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def tag_descriptions(self) -> List[str]:
        """
        Returns one description block per registered tool showing every
        argument in nested-tag format, e.g.:

          write_tool — writes the given text to a file
            <write_tool>
              <arg1>text</arg1>
            </write_tool>

        Used by build_tool_prompt() to auto-generate the system prompt.
        """
        blocks = []
        for name, meta in self._tools.items():
            inner = "\n".join(
                f"  <arg{i+1}>{label}</arg{i+1}>"
                for i, label in enumerate(meta.arg_names)
            )
            blocks.append(
                f"  {name} \u2014 {meta.description}\n"
                f"  <{name}>\n{inner}\n  </{name}>"
            )
        return blocks

    def __contains__(self, name: str) -> bool:
        return name in self._tools
