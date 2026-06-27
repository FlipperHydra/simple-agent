from __future__ import annotations
import asyncio
import ollama
from tool_registry import ToolRegistry
from tool_processor import ToolProcessor
from agent_context import (
    AgentContext,
    TIER_MODEL_MAP,
    TIER_DESCRIPTIONS,
    REQUIRES_CONFIRMATION,
)
from prompts import orchestrator_prompt, FORMAT_PROMPT, REASONING_PROMPT

_client = ollama.AsyncClient()


def _build_confirmed_models() -> set[str]:
    confirmed: set[str] = set()

    for tier, needs_confirm in REQUIRES_CONFIRMATION.items():
        if not needs_confirm:
            continue

        model_name = TIER_MODEL_MAP[tier]
        description = TIER_DESCRIPTIONS[tier]

        warning = (
            f"\n\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\n"
            f"\u2502  \u26a0  Confirmation required: {tier.value.upper()} tier ({model_name})\n"
            f"\u2502\n"
            f"\u2502  {description}\n"
            f"\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518\n"
            f"Allow sub-agents to use {model_name}? (y/n): "
        )

        if input(warning).strip().lower() == "y":
            confirmed.add(model_name)
            print(f"[main] {model_name} approved for sub-agents.\n")
        else:
            print(f"[main] {model_name} restricted to orchestrator only.\n")

    return confirmed


async def main() -> None:
    confirmed_models = _build_confirmed_models()

    orchestrator_context = AgentContext(
        restrictions="",
        confirmed_models=confirmed_models,
    )

    registry = ToolRegistry(context=orchestrator_context)

    system_messages = [
        {"role": "system", "content": orchestrator_prompt(registry)},
        {"role": "system", "content": FORMAT_PROMPT},
        {"role": "system", "content": REASONING_PROMPT},
    ]

    messages = list(system_messages)

    print("Agent ready. Type /clear to reset history, /quit to exit.")

    while True:
        user_message = input("\nYou: ").strip()

        if user_message == "/quit":
            break

        if user_message == "/clear":
            messages = list(system_messages)
            orchestrator_context.orchestrator_brief = ""
            print("[History cleared]")
            continue

        if not user_message:
            continue

        orchestrator_context.orchestrator_brief = user_message
        messages.append({"role": "user", "content": user_message})

        tp = ToolProcessor(registry)

        response = await _client.chat(
            model="gemma4",
            messages=messages,
            think=True,
            stream=True,
        )

        full_response = ""
        in_thinking = False

        async for chunk in response:
            if chunk.message.thinking:
                if not in_thinking:
                    print("\n\u2500\u2500 Thinking \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n")
                    in_thinking = True
                print(chunk.message.thinking, end="", flush=True)

            elif chunk.message.content:
                if in_thinking:
                    print("\n\n\u2500\u2500 Final Answer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n")
                    in_thinking = False
                print(chunk.message.content, end="", flush=True)
                full_response += chunk.message.content
                tp.feed(chunk)

        await tp.finalize()
        print()

        messages.append({"role": "assistant", "content": full_response})

        tool_results = tp.flush_results()
        for tr in tool_results:
            messages.append({
                "role": "user",
                "content": f"[Tool result: {tr['tool']}]\n{tr['result']}"
            })

        if tool_results:
            followup = await _client.chat(
                model="gemma4",
                messages=messages,
                think=True,
                stream=True,
            )

            followup_response = ""
            in_thinking = False
            tp2 = ToolProcessor(registry)

            async for chunk in followup:
                if chunk.message.thinking:
                    if not in_thinking:
                        print("\n\u2500\u2500 Thinking \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n")
                        in_thinking = True
                    print(chunk.message.thinking, end="", flush=True)

                elif chunk.message.content:
                    if in_thinking:
                        print("\n\n\u2500\u2500 Final Answer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n")
                        in_thinking = False
                    print(chunk.message.content, end="", flush=True)
                    followup_response += chunk.message.content
                    tp2.feed(chunk)

            await tp2.finalize()
            print()
            messages.append({"role": "assistant", "content": followup_response})

    print(f"\n[Session complete — {orchestrator_context._spawned_count} sub-agent(s) spawned]")


if __name__ == "__main__":
    asyncio.run(main())
