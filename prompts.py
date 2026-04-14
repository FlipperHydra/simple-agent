from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tool_registry import ToolRegistry


# Dynamic prompt builder


def build_tool_prompt(registry: "ToolRegistry") -> str:
    """
    Generates the TOOL_PROMPT at runtime from the registry so that every
    newly registered tool is automatically documented — no manual edits needed.
    """
    # Tool listing block
    tool_lines = "\n\n".join(registry.tag_descriptions())

    # One example block per tool
    example_blocks = []
    for name, meta in registry.all().items():
        inner = "\n".join(
            f"<arg{i+1}>\nexample {label} here\n</arg{i+1}>"
            for i, label in enumerate(meta.arg_names)
        )
        example_blocks.append(
            f"<{name}>\n{inner}\n</{name}>"
        )
    examples = "\n\n".join(example_blocks)

    # Valid closing tags list for the rules section
    tag_list = ", ".join(f"</{n}>" for n in registry.names())

    return f"""\
You are a helpful assistant. You have access to the following tools:

{tool_lines}

TOOL USAGE FORMAT
─────────────────
When you want to invoke a tool, wrap your arguments in nested tags inside
the tool's own tags:

  <tool_name>
  <arg1>first argument value</arg1>
  <arg2>second argument value</arg2>
  </tool_name>

Rules:
1. The opening tool tag must be on its own line.
2. Each argument uses its own numbered tag: <arg1>, <arg2>, <arg3>, and so on.
3. The argument value goes between its open and close tag.
4. Every <argN> tag must have a matching </argN> close tag.
5. The closing tool tag must be on its own line after all argument tags.
6. No extra text between the outer tool tags — only <argN> blocks.
7. You may call a tool multiple times by repeating the entire block.
8. Write your normal conversational reply outside the tool tags.
9. Valid closing tool tags are: {tag_list}
10. Always include all required arguments for a tool, in order starting from <arg1>.

EXAMPLES
────────
{examples}
"""


#Static prompts, add or remove rules here as needed


FORMAT_PROMPT = """\
CHARACTER RULES — APPLY TO ALL OUTPUT
────────────────────────────────────────────────────────
These rules govern every character you emit: conversational replies,
tool arguments, and any other output.

ALLOWED CHARACTERS
  - Standard ASCII letters and digits: A-Z, a-z, 0-9
  - Basic punctuation: . , ! ? : ; '
  - Spaces and standard newlines (in conversational text only)
  - Escaped newline (\\n) inside tool arguments to represent a line break
FORBIDDEN CHARACTERS
  - Any non-ASCII character (accented letters, symbols, emojis, etc.)
  - Parentheses outside of tool arguments
  - < and > outside of the tool tags and argument tags
  - Dashes, bullets, or newlines as list separators — use letter prefixes instead:
      A. first item, B. second item, C. third item

────────────────────────────────────────────────────────
RULES SPECIFIC TO TOOL ARGUMENTS
────────────────────────────────────────────────────────
  - Use ASCII only. Replace accented characters with their plain ASCII equivalent.
  - Use \\n for line breaks — never insert a literal newline inside an argument.
  - Never place double-quote characters inside an argument.
  - Use single-quotes for any quotation within the text.
"""


REASONING_PROMPT = """\
BEFORE PRODUCING YOUR RESPONSE
────────────────────────────────────────
1. PLAN YOUR TOOL USAGE
   Think through whether a tool call is needed for this response.
   If yes, identify: A. which tool to call, B. how many arguments it needs,
   C. what value to put in each <argN> tag, D. where in the response
   the call belongs.
   Do this silently as part of your reasoning before writing anything.

2. SELF-CHECK AGAINST ALL INSTRUCTIONS
   Before finalizing your response, output your final response FIRST in
   your Thinking stage, verify that every rule given in the
   system prompt is satisfied. Then work through each rule explicitly:
   A. Are all forbidden characters absent from the entire response?
   B. Are all tool blocks formatted correctly with the right tags?
   C. Does every <argN> tag have a matching </argN> close tag?
   D. Are list items separated with letter prefixes, not bullets or dashes?
   E. Are all tool arguments valid ASCII with \\n for line breaks?

   If any check fails, correct the response before outputting it.
   Do not output a response that violates any instruction.
"""