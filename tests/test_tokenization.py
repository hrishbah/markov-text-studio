import pytest
from markov_text.tokenization import tokenize_words, tokenize_chars


def test_tokenize_words_basic():
    assert tokenize_words("hello world") == ["hello", "world"]


def test_tokenize_words_preserves_punctuation():
    assert tokenize_words("Hello, world!") == ["Hello,", "world!"]


def test_tokenize_words_empty():
    assert tokenize_words("") == []


def test_tokenize_words_whitespace_only():
    assert tokenize_words("   \n\t  ") == []


def test_tokenize_words_invalid():
    with pytest.raises(TypeError):
        tokenize_words(123)


def test_tokenize_chars_basic():
    assert tokenize_chars("abc") == ["a", "b", "c"]


def test_tokenize_chars_empty():
    assert tokenize_chars("") == []


def test_tokenize_chars_invalid():
    with pytest.raises(TypeError):
        tokenize_chars(None)
