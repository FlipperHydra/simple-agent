from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tool_registry import ToolRegistry


def tool_prompt(registry: 'ToolRegistry') -> str:
    tool_lines = '\n\n'.join(registry.tag_descriptions())

    example_blocks = []
    for name, meta in registry.all().items():
        if meta.arg_names:
            inner = '\n'.join(
                f'<arg{i+1}>\nexample {label} here\n</arg{i+1}>'
                for i, label in enumerate(meta.arg_names)
            )
            example_blocks.append(
                f'<{name}>\n{inner}\n</{name}>'
            )
        else:
            example_blocks.append(f'<{name}></{name}>')
    examples = '\n\n'.join(example_blocks)

    tag_list = ', '.join(f'</{n}>' for n in registry.names())

    return f"""\
You are a helpful assistant. You have access to the following tools:

{tool_lines}

TOOL USAGE FORMAT
-----------------
When you want to invoke a tool, wrap your arguments in nested tags inside
the tool's own tags:

  <tool_name>
  <arg1>first argument value</arg1>
  <arg2>second argument value</arg2>
  </tool_name>

For tools with no arguments, use an empty tag pair:

  <tool_name></tool_name>

Rules:
1. The opening tool tag must be on its own line.
2. Each argument uses its own numbered tag: <arg1>, <arg2>, <arg3>, and so on.
3. The argument value goes between its open and close tag.
4. Every <argN> tag must have a matching </argN> close tag.
5. The closing tool tag must be on its own line after all argument tags.
6. No extra text between the outer tool tags -- only <argN> blocks.
7. You may call a tool multiple times by repeating the entire block.
8. Write your normal conversational reply outside the tool tags.
9. Valid closing tool tags are: {tag_list}
10. Always include all required arguments for a tool, in order starting from <arg1>.
11. Tools marked [DANGEROUS] have destructive or irreversible effects.
    Before calling a dangerous tool, state clearly what you are about to do
    and why. Never call a dangerous tool speculatively or as a guess.

EXAMPLES
--------
{examples}
"""


def soul_prompt(soul_content: str) -> str:
    return f"""\
AGENT IDENTITY AND CHARACTER
----------------------------------------
The following document defines your identity, voice, values, and
known information about the user you are speaking with.
Read it carefully. It governs how you present yourself in all responses.

{soul_content}

If the User Profile section is empty, treat the user as unknown and
build an impression through conversation. Do not invent facts about
the user that are not in this document.
"""


def soul_update_prompt(memory_content: str, soul_content: str) -> str:
    return f"""\
You are performing a soul update for this agent.

Below is the agent's current soul.md document and the full contents
of memory.md which records timestamped notes about the user across sessions.

CURRENT SOUL.MD:
{soul_content}

MEMORY.MD CONTENTS:
{memory_content}

YOUR TASK:
1. Read through memory.md carefully.
2. Identify patterns in the user's preferred topics, their tone and
   communication style, their recurring goals, and any personal facts
   they have shared.
3. List your insights as letter-prefixed items so the user can review
   them before any changes are committed.
4. Then rewrite ONLY the User Profile section of soul.md with a
   concise, updated summary of the user based on those insights.
5. Output the complete updated soul.md with all other sections unchanged.
6. Use ASCII only. No bullets or dashes -- use letter prefixes for lists.
7. Keep the User Profile section factual and grounded in memory only.
   Do not speculate or invent traits not evidenced in memory.md.
"""


MEMORY_PROMPT = """\
MEMORY TOOLS
----------------------------------------
You have access to two persistent memory tools:

  append_memory -- stores a timestamped note to memory.md.
  recall_memory -- reads all stored notes from memory.md.

When to use these tools:
A. Call append_memory when the user shares something meaningful about
   themselves, their preferences, goals, or context that would be
   useful to remember across sessions.
B. Call append_memory when you complete a significant task the user
   may want to reference or build upon in a future session.
C. Call recall_memory at the start of a session when the user references
   past context that is not present in the current message history.
D. Do not spam memory. Only store genuinely useful, durable information.
E. Do not store trivial, redundant, or one-off notes.
"""


FORMAT_PROMPT = """\
CHARACTER RULES -- APPLY TO ALL OUTPUT
----------------------------------------
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
  - Dashes, bullets, or newlines as list separators -- use letter prefixes instead:
      A. first item, B. second item, C. third item

----------------------------------------
RULES SPECIFIC TO TOOL ARGUMENTS
----------------------------------------
  - Use ASCII only. Replace accented characters with their plain ASCII equivalent.
  - Use \\n for line breaks -- never insert a literal newline inside an argument.
  - Never place double-quote characters inside an argument.
  - Use single-quotes for any quotation within the text.
"""


REASONING_PROMPT = """\
BEFORE PRODUCING YOUR RESPONSE
----------------------------------------
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
