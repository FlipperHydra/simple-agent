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


def test_propose_soul_remove_registered_and_visible():
    # propose_soul_remove must now be a real registered tool: it should appear
    # in the registry, in tag_descriptions(), in tool_prompt()'s tool_lines and
    # EXAMPLES block, and in the ToolProcessor detection pattern.
    from tool_registry import ToolRegistry
    from prompts import tool_prompt
    from tool_processor import ToolProcessor

    reg = ToolRegistry()
    assert 'propose_soul_remove' in reg.names()

    joined = '\n'.join(reg.tag_descriptions())
    assert 'propose_soul_remove' in joined

    tp = tool_prompt(reg)
    assert 'propose_soul_remove' in tp
    example_frag = (
        '<propose_soul_remove>\n<arg1>\nexample section here\n'
        '</arg1>\n</propose_soul_remove>'
    )
    assert example_frag in tp

    proc = ToolProcessor(reg)
    assert 'propose_soul_remove' in proc._pattern.pattern


def test_soul_default_constraints_has_item_e():
    # Item E must exist in both SOUL_DEFAULT (main.py) and the shipped soul.md.
    import main
    assert 'E. Do not narrate routine tool mechanics' in main.SOUL_DEFAULT
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'soul.md')) as f:
        soul = f.read()
    assert 'E. Do not narrate routine tool mechanics' in soul


if __name__ == '__main__':
    test_all_modules_import()
    test_registry_builds_and_lists_tools()
    test_write_tool_replace_is_valid()
    test_propose_soul_remove_registered_and_visible()
    test_soul_default_constraints_has_item_e()
    print('test_smoke: OK')
