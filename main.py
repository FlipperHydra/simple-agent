from __future__ import annotations
import asyncio
import ollama
from tool_registry import ToolRegistry
from tool_processor import ToolProcessor
from prompts import tool_prompt, FORMAT_PROMPT, REASONING_PROMPT

_client = ollama.AsyncClient()


async def main() -> None:
    registry = ToolRegistry()

    system_messages = [
        {'role': 'system', 'content': tool_prompt(registry)},
        {'role': 'system', 'content': FORMAT_PROMPT},
        {'role': 'system', 'content': REASONING_PROMPT},
    ]

    messages = list(system_messages)

    print('Agent ready. Type /clear to reset history, /quit to exit.')

    while True:
        user_message = input('\nYou: ').strip()

        if user_message == '/quit':
            break

        if user_message == '/clear':
            messages = list(system_messages)
            print('[History cleared]')
            continue

        if not user_message:
            continue

        messages.append({'role': 'user', 'content': user_message})

        tp = ToolProcessor(registry)

        response = await _client.chat(
            model='gemma4',
            messages=messages,
            think=True,
            stream=True,
        )

        full_response = ''
        in_thinking = False

        async for chunk in response:
            if chunk.message.thinking:
                if not in_thinking:
                    print('\n-- Thinking --------------------------------------------------\n')
                    in_thinking = True
                print(chunk.message.thinking, end='', flush=True)

            elif chunk.message.content:
                if in_thinking:
                    print('\n\n-- Final Answer -----------------------------------------------\n')
                    in_thinking = False
                print(chunk.message.content, end='', flush=True)
                full_response += chunk.message.content
                tp.feed(chunk)

        await tp.finalize()
        print()

        messages.append({'role': 'assistant', 'content': full_response})

        tool_results = tp.flush_results()
        for tr in tool_results:
            messages.append({
                'role': 'user',
                'content': f"[Tool result: {tr['tool']}]\n{tr['result']}"
            })

        if tool_results:
            followup = await _client.chat(
                model='gemma4',
                messages=messages,
                think=True,
                stream=True,
            )

            followup_response = ''
            in_thinking = False
            tp2 = ToolProcessor(registry)

            async for chunk in followup:
                if chunk.message.thinking:
                    if not in_thinking:
                        print('\n-- Thinking --------------------------------------------------\n')
                        in_thinking = True
                    print(chunk.message.thinking, end='', flush=True)

                elif chunk.message.content:
                    if in_thinking:
                        print('\n\n-- Final Answer -----------------------------------------------\n')
                        in_thinking = False
                    print(chunk.message.content, end='', flush=True)
                    followup_response += chunk.message.content
                    tp2.feed(chunk)

            await tp2.finalize()
            print()
            messages.append({'role': 'assistant', 'content': followup_response})

    print('\n[Session complete]')


if __name__ == '__main__':
    asyncio.run(main())
