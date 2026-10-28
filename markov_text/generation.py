"""Text generation using a Markov model."""

import random
from typing import List, Optional, Tuple
from collections import Counter

from .model import MarkovModel


def generate_text(
    model: MarkovModel,
    length: int,
    seed: Optional[int] = None,
    start: Optional[str] = None,
    mode: str = "word",
) -> str:
    """Generate text from a fitted MarkovModel."""
    if seed is not None:
        random.seed(seed)
    if length < 1:
        raise ValueError("length must be >= 1")
    tokens = list(model.tokens)
    order = model.order

    # Initialize sequence
    if start is not None and start.strip() != "":
        start_tokens = start.split() if mode == "word" else list(start)
        # Trim to order if longer
        if len(start_tokens) > order:
            start_tokens = start_tokens[-order:]
        seq = list(start_tokens)
    else:
        # Pick random start context from model
        contexts = list(model.transitions.keys())
        if not contexts:
            raise ValueError("model has no transitions")
        seq = list(random.choice(contexts))

    # Ensure seq has exactly order items for first step if needed
    if len(seq) < order and tokens:
        # Pad with beginning of corpus for continuity
        seq = tokens[: order - len(seq)] + seq
        # If still too short, just use first tokens
        if len(seq) < order:
            seq = tokens[:order]

    generated = list(seq)
    max_attempts = length * 10 + 100  # guard against infinite loops
    attempts = 0

    while len(generated) < length and attempts < max_attempts:
        attempts += 1
        context = tuple(generated[-order:]) if len(generated) >= order else tuple(generated)
        counts = model.get_counts_for_context(context)
        # If no transition for this context, try shorter context (backoff)
        if not counts and len(context) > 1:
            for shorter in range(len(context) - 1, 0, -1):
                shorter_ctx = context[-shorter:] if shorter < len(context) else context
                counts = model.get_counts_for_context(shorter_ctx)
                if counts:
                    break
        if not counts:
            # Restart with a random context from model
            contexts = list(model.transitions.keys())
            if contexts:
                rand_ctx = random.choice(contexts)
                # Append first token of new context to maintain flow somewhat
                generated.extend(rand_ctx[:1])
            else:
                break
            continue
        choices = list(counts.keys())
        weights = list(counts.values())
        next_token = random.choices(choices, weights=weights, k=1)[0]
        generated.append(next_token)

    result_tokens = generated[:length]
    if mode == "word":
        return " ".join(result_tokens)
    return "".join(result_tokens)
