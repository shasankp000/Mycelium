"""
Tests for the text processing utilities.
"""


from mycelium.pipeline.phase2.utils.text_processors import (
    TextProcessor,
    TokenCounter,
)


class TestTextProcessor:
    def setup_method(self):
        self.tp = TextProcessor()

    def test_clean_html_basic(self):
        result = self.tp.clean_html("<p>Hello</p>")
        assert result == "Hello"

    def test_clean_html_nested(self):
        result = self.tp.clean_html(
            "<div><p>Hello <b>world</b></p></div>"
        )
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_remove_control_chars(self):
        result = self.tp.remove_unicode_control_chars(
            "abc\x00def\x01ghi"
        )
        assert "\x00" not in result
        assert "\x01" not in result
        assert "abcdefghi" in result

    def test_normalize_whitespace(self):
        result = self.tp.normalize_whitespace("  a   b   c  ")
        assert result == "a b c"

    def test_standardize_quotes(self):
        result = self.tp.standardize_quotes(
            "\u201cHello\u201d \u2018world\u2019"
        )
        assert result == '"Hello" \'world\''

    def test_decode_html_entities(self):
        result = self.tp.decode_html_entities("&lt;div&gt;")
        assert result == "<div>"

    def test_fix_encoding_issues(self):
        result = self.tp.fix_encoding_issues("hello world")
        assert isinstance(result, str)
        assert "hello" in result

    def test_full_clean(self):
        raw = "<b>Hello</b> &amp; \x00world\u201c!\u201d"
        result = self.tp.full_clean(raw)
        assert "<" not in result
        assert "\x00" not in result
        assert isinstance(result, str)


class TestTokenCounter:
    def setup_method(self):
        self.tc = TokenCounter()

    def test_token_counting_basic(self):
        assert self.tc.count_tokens("one two three") == 3

    def test_token_counting_empty(self):
        assert self.tc.count_tokens("") == 0

    def test_token_counting_multilingual(self):
        count = self.tc.count_tokens("Hello Welt Bonjour 世界")
        assert count == 4

    def test_estimate_token_count(self):
        estimate = self.tc.estimate_token_count(
            "Hello world this is a test"
        )
        assert estimate > 0

    def test_estimate_token_count_empty(self):
        assert self.tc.estimate_token_count("") == 0

    def test_token_distribution(self):
        dist = self.tc.get_token_distribution("one two three")
        assert dist["total_tokens"] == 3
        assert dist["unique_tokens"] == 3
        assert dist["avg_token_length"] == 3
        assert dist["max_token_length"] == 5

    def test_token_distribution_empty(self):
        dist = self.tc.get_token_distribution("")
        assert dist["total_tokens"] == 0
