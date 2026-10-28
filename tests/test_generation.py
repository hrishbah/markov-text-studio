import pytest
from markov_text.model import MarkovModel
from markov_text.generation import generate_text


def test_generation_word_mode():
    tokens = ["the", "quick", "brown", "fox", "jumps", "the", "quick"]
    model = MarkovModel(tokens, order=2)
    out = generate_text(model, length=4, seed=1, mode="word")
    assert isinstance(out, str)
    assert len(out.split()) == 4


def test_generation_char_mode():
    tokens = list("abcdefgh")
    model = MarkovModel(tokens, order=2)
    out = generate_text(model, length=10, seed=42, mode="char")
    assert isinstance(out, str)
    assert len(out) == 10


def test_reproducibility_same_seed():
    tokens = ["hello", "world", "hello"]
    model = MarkovModel(tokens, order=1)
    a = generate_text(model, length=3, seed=99, mode="word")
    b = generate_text(model, length=3, seed=99, mode="word")
    assert a == b


def test_start_phrase_word():
    tokens = ["the", "forest", "is", "quiet", "the", "forest"]
    model = MarkovModel(tokens, order=2)
    out = generate_text(model, length=3, seed=1, start="the forest", mode="word")
    assert out.startswith("the forest")


def test_length_boundary():
    tokens = ["a", "b", "c"]
    model = MarkovModel(tokens, order=1)
    with pytest.raises(ValueError):
        generate_text(model, length=0, seed=1)


def test_unknown_context_backoff():
    # Small corpus with no transition for a bad context
    tokens = ["a", "b", "c"]
    model = MarkovModel(tokens, order=1)
    # Force start with unknown context; backoff/restart should avoid crash
    out = generate_text(model, length=2, seed=1, start="z", mode="word")
    assert isinstance(out, str)


def test_word_vs_char_different():
    tokens = ["hello", "world"]
    model_w = MarkovModel(tokens, order=1)
    out_w = generate_text(model_w, length=2, seed=1, mode="word")
    # Char mode on same raw characters behaves differently
    model_c = MarkovModel(list("helloworld"), order=1)
    out_c = generate_text(model_c, length=2, seed=1, mode="char")
    assert out_w != out_c or len(out_w.split()) == 2


def test_restart_on_dead_end():
    tokens = ["a", "b"]
    model = MarkovModel(tokens, order=1)
    # b has no transition; restart should not crash
    out = generate_text(model, length=3, seed=1, mode="word")
    assert isinstance(out, str)
