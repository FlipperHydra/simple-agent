from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tool_registry import ToolRegistry


def tool_prompt(registry: "ToolRegistry") -> str:
    tool_lines = "\n\n".join(registry.tag_descriptions())

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

    tag_list = ", ".join(f"</{n}>" for n in registry.names())

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

EXAMPLES
--------
{examples}
"""


subagent_prompt = (
    "You are a sub-agent. You have been assigned a specific task by an orchestrator.\n"
    "Your role is to complete that task and return a clear, concise result.\n\n"
    "Rules:\n"
    "- Focus only on the task you were given. Do not expand scope.\n"
    "- Do not re-plan or re-decompose unless the task explicitly requires it.\n"
    "- When your task is complete, state your result clearly and stop.\n"
    "- Do not keep calling tools after the task is done.\n"
    "- If you cannot complete the task with the tools available, say so clearly."
)


tier_guidance = (
    "When spawning a sub-agent, set the tier argument to one of:\n\n"
    "  light:    phi3-mini -- simple single-step tasks: short text generation,\n"
    "            formatting, basic lookups. Fastest, lowest resource.\n"
    "            Risk: may fail on complex reasoning.\n\n"
    "  standard: qwen2.5:3b -- moderate tasks: multi-step reasoning, code generation,\n"
    "            summarisation. Balanced speed and capability.\n"
    "            Recommended default for most sub-tasks.\n\n"
    "  heavy:    gemma4 -- ONLY for tasks requiring deep reasoning where standard\n"
    "            has failed or is clearly insufficient.\n"
    "            WARNING: high resource usage, significant speed reduction.\n"
    "            Requires explicit user confirmation before use.\n\n"
    "Default to standard unless you have a clear reason to deviate.\n"
    "Never choose heavy unless the task genuinely demands it."
)


def orchestrator_prompt(registry: "ToolRegistry") -> str:
    tool_block = "\n\n".join(registry.tag_descriptions())
    return (
        "You are the orchestrator. You are a capable agent in your own right --\n"
        "you handle tasks directly unless there is a clear reason to delegate.\n\n"
        "Delegate to sub-agents ONLY when a task has real, discernible parallel\n"
        "or independent segments that would genuinely benefit from separation.\n\n"
        "Delegate when:\n"
        "  - The task contains multiple independent components that can be worked\n"
        "    on separately (e.g. building a frontend and a backend simultaneously)\n"
        "  - The task involves processing multiple large, distinct inputs where\n"
        "    each input can be handled in isolation (e.g. analysing several large files)\n"
        "  - The user explicitly asks for parallel or delegated work\n"
        "  - A segment requires a different model capability than the rest of the task\n\n"
        "Do NOT delegate when:\n"
        "  - The task is a single cohesive action, even if it takes effort\n"
        "    (e.g. formatting a document, writing a summary, answering a question)\n"
        "  - Breaking the task up would create more coordination overhead than value\n"
        "  - The task is sequential -- each step depends on the previous result\n"
        "  - You are unsure -- when in doubt, handle it yourself\n\n"
        "When you do delegate, always provide:\n"
        "  - task:          A specific, self-contained sub-task\n"
        "  - context_brief: The overall goal so the sub-agent understands its role\n"
        "  - tier:          The model tier appropriate for this sub-task\n\n"
        "Available tools:\n\n"
        f"{tool_block}\n\n"
        f"{tier_guidance}\n\n"
        "When sub-agents return results, synthesise them into a single coherent\n"
        "answer. Do not return raw sub-agent output -- integrate and present it cleanly."
    )


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

SOUL EDITING RULES
----------------------------------------
You have two tools for modifying your soul.md document:

  propose_soul_edit(section, proposed_content)
    section          -- the exact ## heading name to update, e.g. 'User Profile'
    proposed_content -- the new content to add or the replacement text

  propose_soul_remove(section)
    section          -- the exact ## heading name to remove entirely

SECTION TYPES
Soul sections fall into two types that determine how edits are handled:

List sections: Values, Constraints, User Profile
  These sections contain letter-prefixed entries such as A., B., C.
  For these sections, write only the NEW entry to add -- do not include
  the full section or existing entries. The system assigns the next
  letter automatically. Do not prefix your entry with a letter.
  Example: 'Do not use sycophantic phrasing in any response.'

Prose sections: Identity, Voice and Tone
  These sections contain flowing paragraphs.
  For these sections, write the complete replacement text for the section.

WHEN TO CALL propose_soul_edit
A. The user modifies your behavior or workflow.
   Sections: Constraints or Values
B. The user describes themselves -- location, occupation, habits, goals.
   Section: User Profile
C. The user corrects your tone or communication style.
   Sections: Voice and Tone -- and Values if the correction reflects a principle
D. The user gives you a name.
   Section: Identity
E. The user states a durable preference about how you should work.
   Section: User Profile -- and Constraints if it is a behavioral rule

WHEN TO CALL propose_soul_remove
Call this tool when the user explicitly asks to remove or delete a section
from soul.md entirely. Do not call it speculatively.

RULES FOR PROPOSING EDITS
1. If a single statement affects multiple sections, call propose_soul_edit
   once per affected section.
2. Propose only on clear, durable signals.
3. Do not re-propose a change the user rejected earlier in this session.
4. For list sections, write only the new entry text -- no letter prefix.
5. User Profile entries should be written as factual statements.
6. For Constraints and Values entries, write as behavioral rules using
   the same imperative style as existing entries.
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


def soul_edit_proposal_display(section: str, proposed_content: str, existing_content: str = '') -> str:
    if existing_content and not existing_content.startswith('(No profile'):
        existing_block = f'Existing :\n{existing_content}\n\n'
    else:
        existing_block = 'Existing : (empty)\n\n'

    return (
        f'\n-- Soul Edit Proposed --------------------------------------\n'
        f'Section  : {section}\n'
        f'{existing_block}'
        f'Proposed : {proposed_content}\n'
        f'------------------------------------------------------------\n'
        f'Accept this change? [y/N]: '
    )


def soul_remove_proposal_display(section: str, existing_content: str = '') -> str:
    if existing_content:
        existing_block = f'\nContent that will be removed:\n{existing_content}\n'
    else:
        existing_block = '\n(Section appears to be empty or not found.)\n'

    return (
        f'\n-- Soul Remove Proposed ------------------------------------\n'
        f'Section  : {section}{existing_block}'
        f'------------------------------------------------------------\n'
        f'Remove this section entirely? [y/N]: '
    )


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


from research_prompt import RESEARCH_PROMPT
