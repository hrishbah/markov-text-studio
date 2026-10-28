# Markov Text Studio

Statistical text generation with Markov chains — Prodigy InfoTech Generative AI Internship Task 03.

## Overview

Markov Text Studio builds a statistical transition table from a text corpus and generates new sequences by sampling from observed next-token frequencies. It supports both word-level and character-level generation, with an adjustable Markov order (1–5), optional starting phrase, reproducible random seed, and graded validation for empty or oversized corpora.

## How it works

- Tokenize the corpus (words or characters).
- Count occurrences of each context (previous `n` tokens) followed by each next token.
- At generation time, use the current context to look up counts and sample proportionally.
- Higher orders create more locally coherent sequences but reduce transition counts for small corpora.
- Unknown contexts trigger shorter-context backoff or a restart from a random observed context.

## Project tree

```
.
├── app.py                     Streamlit application
├── data/
│   └── sample_corpus.txt      Original built-in corpus
├── markov_text/
│   ├── __init__.py
│   ├── tokenization.py        Word / char tokenizers
│   ├── model.py               Transition table builder
│   └── generation.py          Sampling generator
├── outputs/
│   └── .gitkeep               Kept; generated .txt ignored
├── tests/
│   ├── test_tokenization.py
│   ├── test_model.py
│   └── test_generation.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python 3.12
- `streamlit==1.66.1`
- `pytest==9.1.1`

No PyTorch, NumPy, Transformers, or external APIs are used.

## Setup

```bash
cd prodigy-genai-task-03-markov-text-generation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Tests:

```bash
python -m pytest -q
```

## Features

- Corpus sources: built-in original sample, pasted text, or uploaded `.txt`.
- Mode: word-level (preserves punctuation spacing) or character-level.
- Markov order: 1–5, default 2.
- Optional starting phrase.
- Seeded reproducible output.
- Corpus stats shown after training.
- Download generated text and save locally when requested.
- Graceful handling of very short or repetitive corpora.
- 1 MB input-size limit with clear rejection messages.

## Use cases

- Generate whimsical prose variations from a small story.
- Experiment with order and mode to observe coherence vs. diversity tradeoffs.
- Compare word-level patterns with character-level patterns using the same seed.

## Limitations

- Output is purely statistical; it does not understand semantics, grammar, or context.
- Very small or highly repetitive corpora may produce repetitive or broken output.
- Character mode can generate whitespace and punctuation unexpectedly.

## Verified results

- `python -m pytest -q`: 25 passed.
- `python -m compileall -q app.py markov_text tests`: clean.
- `python -m pip check`: no broken requirements.
- `python -c "import markov_text; print(markov_text.__file__)"`: passes.
- Streamlit starts at `http://localhost:3082` (or default) without auto-generating text.
- Real word and character generation verified with `data/sample_corpus.txt`.

## Task attribution

Prodigy InfoTech Generative AI Internship — Task 03: Text Generation with Markov Chains.
