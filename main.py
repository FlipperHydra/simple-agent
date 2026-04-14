from ollama import chat
from tool_registry import ToolRegistry
from tool_processor import ToolProcessor
from prompts import build_tool_prompt, FORMAT_PROMPT, REASONING_PROMPT


def main() -> None:

    registry = ToolRegistry()
    tp = ToolProcessor(registry)

    # Build the tool prompt dynamically — picks up every registered tool
    tool_prompt = build_tool_prompt(registry)

    user_message = input("Enter your message: ")
    # The 'model' aspect of the system.
    # Change model and content as required.
    response = chat(
        model='gemma4',
        messages=[
            {'role': 'system', 'content': tool_prompt},
            {'role': 'system', 'content': FORMAT_PROMPT},
            {'role': 'system', 'content': REASONING_PROMPT},
            {'role': 'user',   'content': user_message}
        ],
        think=True,
        stream=True,
    )

    in_thinking = False
    # Thinking stream and content stream are interleaved, so we check each chunk for both.
    for chunk in response:
        if chunk.message.thinking:
            if not in_thinking:
                print("\n── Thinking ──────────────────────────────────\n")
                in_thinking = True
            print(chunk.message.thinking, end='', flush=True)

        elif chunk.message.content:
            if in_thinking:
                print("\n\n── Final Answer ──────────────────────────────\n")
                in_thinking = False
            print(chunk.message.content, end='', flush=True)
            tp.feed(chunk)

    tp.finalize()
    print()


if __name__ == "__main__":
    main()