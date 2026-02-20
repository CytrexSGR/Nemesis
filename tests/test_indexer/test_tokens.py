"""Tests fuer nemesis.indexer.tokens."""

from nemesis.indexer.tokens import count_tokens, estimate_tokens


class TestCountTokens:
    def test_count_tokens_simple(self):
        result = count_tokens("Hello, world!")
        assert isinstance(result, int)
        assert result > 0

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_count_tokens_whitespace_only(self):
        result = count_tokens("   \n\t  ")
        assert isinstance(result, int)
        assert result >= 0

    def test_count_tokens_long_code(self):
        code = "\n".join(f"x_{i} = {i}" for i in range(200))
        result = count_tokens(code)
        assert result > 100

    def test_count_tokens_multiline_function(self):
        code = (
            "def fibonacci(n: int) -> int:\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n"
        )
        result = count_tokens(code)
        assert 30 <= result <= 200


class TestEstimateTokens:
    def test_estimate_tokens_approximation(self):
        result = estimate_tokens("This is a test string for estimation.")
        assert isinstance(result, int)
        assert result > 0

    def test_estimate_vs_count_same_order_of_magnitude(self):
        text = "def foo(bar, baz): return bar + baz"
        estimated = estimate_tokens(text)
        counted = count_tokens(text)
        assert estimated > 0
        assert counted > 0
        ratio = max(estimated, counted) / max(min(estimated, counted), 1)
        assert ratio <= 3.0
