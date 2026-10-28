"""Tokenization utilities for word-level and character-level modes."""

from typing import List


def tokenize_words(text: str) -> List[str]:
    """Split text into word tokens preserving punctuation spacing."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    # Simple word split that keeps punctuation attached to words
    return text.split()


def tokenize_chars(text: str) -> List[str]:
    """Split text into individual characters."""
    if not isinstance(text, str):
        raise TypeError("text must be str")
    return list(text)
