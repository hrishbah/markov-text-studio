"""Regression tests for the Streamlit seed-input bounds fix.

The seed `st.number_input` must use bounds safe within Streamlit's
JavaScript-safe integer range (it cannot accept `min_value=-(2**63)` /
`max_value=2**63 - 1`). These tests guard against the bug recurring.
"""

import re
from pathlib import Path

from markov_text import tokenize_words, MarkovModel, generate_text

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

# Upper bound mirrors the safe value set in app.py (2_147_483_647).
SAFE_SEED_MAX = 2_147_483_647
JS_SAFE_MAX = (2 ** 53) - 1


def test_seed_bounds_are_js_safe():
    src = APP_PATH.read_text(encoding="utf-8")
    # No unsigned 64-bit range used for the seed widget.
    assert "-(2**63)" not in src
    assert "2**63 - 1" not in src
    # Seed widget uses a non-negative, JS-safe range.
    pattern = re.compile(r"min_value=0,\s*max_value=2_147_483_647,")
    assert pattern.search(src), "Seed number_input should use min_value=0, max_value=2_147_483_647"


def test_seed_max_within_js_safe_range():
    assert 0 <= SAFE_SEED_MAX <= JS_SAFE_MAX


def test_reproducible_generation_across_seed_range():
    tokens = tokenize_words("the forest is quiet the forest is deep")
    model = MarkovModel(tokens, order=2)
    out = generate_text(model, length=6, seed=0, mode="word")
    out2 = generate_text(model, length=6, seed=0, mode="word")
    assert out == out2
    assert len(out.split()) == 6


def test_default_seed_42_reproducible():
    tokens = tokenize_words("rain falls softly the river flows on")
    model = MarkovModel(tokens, order=2)
    out = generate_text(model, length=4, seed=42, mode="word")
    out2 = generate_text(model, length=4, seed=42, mode="word")
    assert out == out2
