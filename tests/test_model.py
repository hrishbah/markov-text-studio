import pytest
from markov_text.model import MarkovModel
from collections import Counter


def test_model_build_and_counts():
    tokens = ["a", "b", "a", "c"]
    model = MarkovModel(tokens, order=1)
    counts = model.get_counts_for_context(("a",))
    assert counts == Counter({"b": 1, "c": 1})


def test_model_order_2():
    tokens = ["the", "cat", "sat", "the", "cat", "slept"]
    model = MarkovModel(tokens, order=2)
    counts = model.get_counts_for_context(("the", "cat"))
    assert counts == Counter({"sat": 1, "slept": 1})


def test_model_empty_tokens_raises():
    with pytest.raises(ValueError):
        MarkovModel([])


def test_model_invalid_order():
    with pytest.raises(ValueError):
        MarkovModel(["a"], order=0)
    with pytest.raises(ValueError):
        MarkovModel(["a"], order=10)


def test_total_transitions():
    tokens = ["a", "b", "c", "a", "b"]
    model = MarkovModel(tokens, order=1)
    total = model.total_transitions()
    # a->b, b->c, c->a, a->b = 4
    assert total == 4
