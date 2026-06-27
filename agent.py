from __future__ import annotations
import ollama
from tool_registry import ToolRegistry
from tool_processor import ToolProcessor
from prompts import tool_prompt, subagent_prompt, FORMAT_PROMPT, REASONING_PROMPT
from agent_context import AgentContext

_client = ollama.AsyncClient()


async def run_agent(
    task: str,
    registry: ToolRegistry,
    model: str = "qwen2.5:3b",
    max_iterations: int = 5,
    tool_filter: list[str] | None = None,
    context: AgentContext | None = None,
) -> str:
    ctx = context or AgentContext()

    if tool_filter is not None:
        filtered_registry = ToolRegistry.__new__(ToolRegistry)
        filtered_registry._tools = {
            name: meta
            for name, meta in registry.all().items()
            if name in tool_filter
        }
    else:
        filtered_registry = registry

    context_block = ""
    if ctx.orchestrator_brief:
        context_block += f"\nOverall goal from orchestrator: {ctx.orchestrator_brief}"
    if ctx.restrictions:
        context_block += f"\nRestrictions you must follow: {ctx.restrictions}"
    if context_block:
        context_block = f"\n[Sub-agent context]{context_block}\n"

    conversation: list[dict] = [
        {"role": "system", "content": tool_prompt(filtered_registry)},
        {"role": "system", "content": subagent_prompt},
        {"role": "system", "content": FORMAT_PROMPT},
        {"role": "system", "content": REASONING_PROMPT},
    ]
    if context_block:
        conversation.append({"role": "system", "content": context_block})
    conversation.append({"role": "user", "content": task})

    final_output = ""

    for iteration in range(max_iterations):
        tp = ToolProcessor(filtered_registry)

        response = await _client.chat(
            model=model,
            messages=conversation,
            think=True,
            stream=True,
        )

        in_thinking = False
        assistant_text = ""

        async for chunk in response:
            if chunk.message.thinking:
                if not in_thinking:
                    print(f"\n\u2500\u2500 [Sub-agent depth={ctx._current_depth}] Thinking (iter {iteration + 1}) \u2500\u2500\n")
                    in_thinking = True
                print(chunk.message.thinking, end="", flush=True)

            elif chunk.message.content:
                if in_thinking:
                    print(f"\n\n\u2500\u2500 [Sub-agent depth={ctx._current_depth}] Response (iter {iteration + 1}) \u2500\u2500\n")
                    in_thinking = False
                print(chunk.message.content, end="", flush=True)
                assistant_text += chunk.message.content
                tp.feed(chunk)

        print()
        final_output = assistant_text
        await tp.finalize()
        tool_results = tp.flush_results()

        conversation.append({"role": "assistant", "content": assistant_text})

        if not tool_results:
            break

        for tr in tool_results:
            conversation.append({
                "role": "user",
                "content": f"[Tool result: {tr['tool']}]\n{tr['result']}"
            })

    return final_output
