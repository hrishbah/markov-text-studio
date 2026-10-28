"""Markov model: transition counts and probability estimation."""

from typing import Dict, List, Tuple
from collections import Counter, defaultdict


class MarkovModel:
    """Builds transition tables from a token sequence."""

    def __init__(self, tokens: List[str], order: int = 2):
        if not tokens:
            raise ValueError("tokens cannot be empty")
        if order < 1 or order > 5:
            raise ValueError(f"order must be 1-5, got {order}")
        self.tokens = tokens
        self.order = order
        self.transitions: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self._build()

    def _build(self) -> None:
        n = self.order
        tokens = self.tokens
        for i in range(len(tokens) - n):
            context = tuple(tokens[i : i + n])
            next_token = tokens[i + n]
            self.transitions[context][next_token] += 1

    def get_counts_for_context(self, context: Tuple[str, ...]) -> Counter:
        return self.transitions.get(context, Counter())

    def total_transitions(self) -> int:
        return sum(sum(c.values()) for c in self.transitions.values())
