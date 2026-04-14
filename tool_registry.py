from typing import Any, Callable, Dict, List, NamedTuple


#Tools 
def save_tool(filename: str, content: str) -> None:
    with open(filename, "a", encoding="utf-8") as f:
        f.write(content + "\n")
    print(f'\n[save_tool] Saved → {filename}')


def write_tool(text: str) -> None:
    formatted = (
        text
        .replace('\\n', '\n')
        .replace('\\t', '\t')
        .replace('\\\'', '\'')
    )
    with open("output.txt", "a", encoding="utf-8") as f:
        f.write(formatted + "\n")
    print(f'\n[write_tool] Wrote → {formatted}')


# Multi-argument example (uncomment to use):
#
# def save_tool(filename: str, content: str) -> None:
#     with open(filename, "a", encoding="utf-8") as f:
#         f.write(content + "\n")
#     print(f'\n[save_tool] Saved → {filename}')


#Tool metadata


class ToolMeta(NamedTuple):
    func: Callable[..., Any]
    arg_names: List[str]   # ordered list of argument labels shown in the prompt
    description: str       # one-line description shown in the prompt


#Tool Registry


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolMeta] = {}
        self._register_defaults()

    #MODIFY THIS TO ADD TOOLS
    def _register_defaults(self) -> None:
        self.register_tool(
            "write_tool",
            write_tool,
            arg_names=["text"],
            description="writes the given text to one file",
        )
        self.register_tool(
            "save_tool",
            save_tool,
            arg_names=["filename", "content"],
            description="saves content to a named file",
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
                f"  {name} — {meta.description}\n"
                f"  <{name}>\n{inner}\n  </{name}>"
            )
        return blocks

    def __contains__(self, name: str) -> bool:
        return name in self._tools