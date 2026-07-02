"""Tests for fetch_url SSRF/scheme blocking and eval_math DoS bound."""
import os

os.environ.setdefault('AGENT_DATA_DIR', '/tmp/simple_agent_test_data')

from tools import Tools, _is_blocked_host


def test_fetch_url_rejects_non_http_schemes():
    assert 'Blocked' in Tools.fetch_url('file:///etc/passwd')
    assert 'Blocked' in Tools.fetch_url('ftp://example.com/x')


def test_fetch_url_blocks_private_and_loopback():
    assert 'Blocked' in Tools.fetch_url('http://127.0.0.1/')
    assert 'Blocked' in Tools.fetch_url('http://10.0.0.1/')
    assert 'Blocked' in Tools.fetch_url('http://169.254.169.254/latest/meta-data/')


def test_is_blocked_host_direct():
    assert _is_blocked_host('localhost') is True
    assert _is_blocked_host('127.0.0.1') is True
    assert _is_blocked_host('192.168.1.1') is True
    assert _is_blocked_host('::1') is True


def test_eval_math_allows_small_powers():
    assert Tools.eval_math('2 ** 10') == '1024'
    assert Tools.eval_math('(3 + 4) * 2') == '14'


def test_eval_math_blocks_large_and_computed_exponents():
    assert 'too large' in Tools.eval_math('10 ** 5000')
    # Nested power (computed exponent, e.g. the classic 9**9**9) is refused.
    assert 'too large or complex' in Tools.eval_math('9 ** 9 ** 9')


def test_eval_math_blocks_disallowed_nodes():
    assert 'Disallowed' in Tools.eval_math('__import__("os")')


if __name__ == '__main__':
    test_fetch_url_rejects_non_http_schemes()
    test_fetch_url_blocks_private_and_loopback()
    test_is_blocked_host_direct()
    test_eval_math_allows_small_powers()
    test_eval_math_blocks_large_and_computed_exponents()
    test_eval_math_blocks_disallowed_nodes()
    print('test_security: OK')
