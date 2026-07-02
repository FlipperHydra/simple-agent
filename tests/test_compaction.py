"""Tests for token counting, the compaction split, persistence, and the
re-prompt-delta decision that drives context compaction."""
import os

os.environ.setdefault('AGENT_DATA_DIR', '/tmp/simple_agent_test_data')

import main
import config


def _convo(n_pairs):
    conv = []
    for i in range(n_pairs):
        conv.append({'role': 'user', 'content': f'user message number {i} ' * 20})
        conv.append({'role': 'assistant', 'content': f'assistant reply number {i} ' * 20})
    return conv


def test_count_tokens_grows_with_content():
    small = [{'role': 'user', 'content': 'hi'}]
    big = [{'role': 'user', 'content': 'word ' * 500}]
    assert main._count_tokens(small) < main._count_tokens(big)
    assert main._count_tokens('') >= 0


def test_split_needs_at_least_two_turns():
    assert main._split_for_compaction([]) is None
    one = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'yo'}]
    assert main._split_for_compaction(one) is None


def test_split_lands_on_user_boundary_and_keeps_recent():
    conv = _convo(6)
    split = main._split_for_compaction(conv)
    assert split is not None
    # Split must be a user-message boundary (never tears a pair apart).
    assert conv[split]['role'] == 'user'
    # Older segment non-empty, recent segment keeps at least the final turn.
    assert 0 < split < len(conv)
    older, recent = conv[:split], conv[split:]
    assert older and recent
    assert recent[0]['role'] == 'user'


def test_reprompt_delta_policy():
    # Mirrors the soft-threshold gate in _maybe_compact: only prompt when over
    # the threshold AND grown beyond the delta since the last decision.
    threshold = main.COMPACT_THRESHOLD_TOKENS
    delta = main.COMPACT_REPROMPT_DELTA

    def should_prompt(total, last):
        return total > threshold and (total - last) > delta

    assert should_prompt(threshold + delta + 1, 0) is True      # first crossing
    assert should_prompt(threshold + 10, threshold + 5) is False  # tiny growth after decision
    assert should_prompt(threshold + delta + 100, threshold) is True
    assert should_prompt(threshold - 1, 0) is False              # under threshold


def test_conversation_persistence_roundtrip(tmp_path=None):
    conv = _convo(2)
    main._save_conversation(conv)
    loaded = main._load_conversation()
    assert loaded == conv


def test_atomic_write_roundtrip():
    p = config.data_path('atomic_probe.txt')
    config.atomic_write(p, 'hello atomic')
    with open(p, encoding='utf-8') as f:
        assert f.read() == 'hello atomic'
    os.remove(p)


if __name__ == '__main__':
    test_count_tokens_grows_with_content()
    test_split_needs_at_least_two_turns()
    test_split_lands_on_user_boundary_and_keeps_recent()
    test_reprompt_delta_policy()
    test_conversation_persistence_roundtrip()
    test_atomic_write_roundtrip()
    print('test_compaction: OK')
