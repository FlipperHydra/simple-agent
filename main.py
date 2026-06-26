from ollama import chat
from tool_registry import ToolRegistry
from tool_processor import ToolProcessor
from prompts import build_tool_prompt, FORMAT_PROMPT, REASONING_PROMPT


def main() -> None:

    registry = ToolRegistry()
    tp = ToolProcessor(registry)

    # Build the tool prompt dynamically — picks up every registered tool
    tool_prompt = build_tool_prompt(registry)

    # System prompts stay fixed across all turns
    messages = [
        {'role': 'system', 'content': tool_prompt},
        {'role': 'system', 'content': FORMAT_PROMPT},
        {'role': 'system', 'content': REASONING_PROMPT},
    ]

    print("Agent ready. Type /clear to reset history, /quit to exit.")

    while True:
        user_message = input("\nYou: ").strip()

        if user_message == "/quit":
            break

        if user_message == "/clear":
            messages = messages[:3]  # keep only the 3 system prompts
            print("[History cleared]")
            continue

        if not user_message:
            continue

        messages.append({'role': 'user', 'content': user_message})

        # The 'model' aspect of the system.
        # Change model and content as required.
        response = chat(
            model='gemma4',
            messages=messages,
            think=True,
            stream=True,
        )

        full_response = ""
        in_thinking = False

        # Thinking stream and content stream are interleaved, so we check each chunk for both.
        for chunk in response:
            if chunk.message.thinking:
                if not in_thinking:
                    print("\n\u2500\u2500 Thinking \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n")
                    in_thinking = True
                print(chunk.message.thinking, end='', flush=True)

            elif chunk.message.content:
                if in_thinking:
                    print("\n\n\u2500\u2500 Final Answer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n")
                    in_thinking = False
                print(chunk.message.content, end='', flush=True)
                full_response += chunk.message.content
                tp.feed(chunk)

        tp.finalize()
        print()

        # Append the assistant's full response to history
        messages.append({'role': 'assistant', 'content': full_response})

        # Inject any tool results back into the conversation so the model can act on them
        tool_results = tp.flush_results()
        for tr in tool_results:
            messages.append({
                'role': 'user',
                'content': f"[Tool result: {tr['tool']}]\n{tr['result']}"
            })

        if tool_results:
            # Let the model reason about the tool results it just received
            followup = chat(
                model='gemma4',
                messages=messages,
                think=True,
                stream=True,
            )

            followup_response = ""
            in_thinking = False

            for chunk in followup:
                if chunk.message.thinking:
                    if not in_thinking:
                        print("\n\u2500\u2500 Thinking \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n")
                        in_thinking = True
                    print(chunk.message.thinking, end='', flush=True)

                elif chunk.message.content:
                    if in_thinking:
                        print("\n\n\u2500\u2500 Final Answer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n")
                        in_thinking = False
                    print(chunk.message.content, end='', flush=True)
                    followup_response += chunk.message.content
                    tp.feed(chunk)

            tp.finalize()
            print()

            messages.append({'role': 'assistant', 'content': followup_response})


if __name__ == "__main__":
    main()
