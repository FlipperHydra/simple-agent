"""Import smoke test.

This would have caught the original SyntaxError in tools.py (a malformed
string literal that broke `from tools import Tools` and therefore the entire
program). If any module fails to import, this test fails.
"""
import importlib
import os

# Isolate any file writes performed at import time to a throwaway dir.
os.environ.setdefault('AGENT_DATA_DIR', '/tmp/simple_agent_test_data')


def test_all_modules_import():
    for name in ('config', 'tools', 'prompts', 'tool_registry', 'tool_processor', 'main'):
        mod = importlib.import_module(name)
        assert mod is not None


def test_registry_builds_and_lists_tools():
    from tool_registry import ToolRegistry
    reg = ToolRegistry()
    names = reg.names()
    assert 'write_tool' in names
    assert 'fetch_url' in names
    assert 'eval_math' in names


def test_write_tool_replace_is_valid():
    # The original bug lived here: the escaped-quote replacement must run
    # without raising (previously a SyntaxError, then a broken .replace call).
    from tools import Tools
    Tools.write_tool("line one\\nwith an escaped quote: it\\'s fine")


if __name__ == '__main__':
    test_all_modules_import()
    test_registry_builds_and_lists_tools()
    test_write_tool_replace_is_valid()
    print('test_smoke: OK')
