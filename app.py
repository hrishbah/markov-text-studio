"""Markov Text Studio — Streamlit interactive text generation app."""

from pathlib import Path
import streamlit as st

from markov_text import tokenize_words, tokenize_chars, MarkovModel, generate_text

ROOT = Path(__file__).resolve().parent
SAMPLE_CORPUS_PATH = ROOT / "data" / "sample_corpus.txt"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_MODES = ["word", "char"]
MAX_INPUT_BYTES = 1024 * 1024

st.set_page_config(
    page_title="Markov Text Studio",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_corpus_text(source_option: str, paste_text: str, uploaded_file) -> str:
    if source_option == "Built-in sample":
        return SAMPLE_CORPUS_PATH.read_text()
    if uploaded_file is not None:
        data = uploaded_file.getvalue()
        text = data.decode("utf-8", errors="replace")
        if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
            raise ValueError("Uploaded file exceeds 1 MB limit.")
        return text
    text = paste_text or ""
    if not text.strip():
        # fallback to sample if custom empty
        return SAMPLE_CORPUS_PATH.read_text()
    if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("Pasted text exceeds 1 MB limit.")
    return text


def main():
    # Title
    st.title("🎭 Markov Text Studio")
    st.caption("Statistical text generation with Markov chains")

    # --- Corpus selection -------------------------------------------------
    with st.container(border=False):
        st.markdown("#### Corpus input")
        source_col, paste_col = st.columns([1, 2])
        with source_col:
            source_option = st.selectbox(
                "Corpus source",
                ["Built-in sample", "Paste custom text", "Upload .txt file"],
                index=0,
                help="Built-in sample is original and varied. Upload is local-only.",
            )
        with paste_col:
            if source_option == "Paste custom text":
                custom_paste = st.text_area(
                    "Pasted corpus",
                    value=SAMPLE_CORPUS_PATH.read_text(),
                    height=160,
                    help="Paste your own text. Must be non-empty and under 1 MB.",
                )
            else:
                custom_paste = ""
        upload_file = None
        if source_option == "Upload .txt file":
            upload_file = st.file_uploader(
                "Upload .txt",
                type=["txt"],
                help="File is read in memory only; never uploaded anywhere else.",
            )

        try:
            corpus = load_corpus_text(source_option, custom_paste, upload_file)
        except ValueError as exc:
            st.error(str(exc))
            corpus = SAMPLE_CORPUS_PATH.read_text()

        # Validation messages
        stripped = corpus.strip()
        if not stripped:
            st.warning("Corpus is blank. Falling back to built-in sample.")
            corpus = SAMPLE_CORPUS_PATH.read_text()
        else:
            byte_len = len(corpus.encode("utf-8"))
            if byte_len < 10:
                st.warning("Corpus is very short; transitions may be sparse.")
            elif byte_len > MAX_INPUT_BYTES:
                st.error("Corpus exceeds 1 MB; using truncated view may cause errors.")

        # Stats preview
        word_tokens = tokenize_words(corpus)
        char_tokens = tokenize_chars(corpus)
        word_uniq = len(set(word_tokens))
        char_uniq = len(set(char_tokens))

    # --- Settings --------------------------------------------------------
    with st.container(border=False):
        st.markdown("#### Generation settings")
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            mode = st.selectbox("Mode", VALID_MODES, index=0, help="Word-level keeps punctuation spacing; char-level writes characters.")
            order = st.slider("Markov order", 1, 5, 2, help="Higher = more context, but fewer observed transitions on small corpora.")
        with s_col2:
            length = st.number_input("Length", min_value=1, max_value=2000, value=10, step=1, help="In words for word mode, characters for char mode.")
            start_phrase = st.text_input("Optional starting phrase", value="", help="Must match mode tokenization.")
        with s_col3:
            seed_value = st.number_input(
                "Random seed (optional)",
                min_value=0, max_value=2_147_483_647,
                value=42,
                step=1,
                help="Same seed + same settings = identical output.",
            )
            seed_enabled = st.checkbox("Use seed for reproducibility", value=True, help="Uncheck to allow random output.")

    # --- Model training ---------------------------------------------------
    with st.container(border=False):
        try:
            tokens = tokenize_words(corpus) if mode == "word" else tokenize_chars(corpus)
            if len(tokens) < order + 1:
                st.warning(f"Corpus too short for order={order}; reducing order.")
                effective_order = max(1, len(tokens) - 1)
            else:
                effective_order = order
            model = MarkovModel(tokens, order=effective_order)
            transition_count = model.total_transitions()
        except Exception as e:
            st.error(f"Model build error: {e}")
            st.stop()

        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        stats_col1.metric("Tokens", len(tokens))
        stats_col2.metric("Unique", len(set(tokens)))
        stats_col3.metric("Transitions", transition_count)
        stats_col4.metric("Order", effective_order)

    # --- Generate --------------------------------------------------------
    with st.container(border=False):
        generate_clicked = st.button("Generate", type="primary", use_container_width=True)

    if generate_clicked:
        try:
            seed_to_use = int(seed_value) if seed_enabled else None
            generated = generate_text(
                model,
                length=int(length),
                seed=seed_to_use,
                start=start_phrase.strip() or None,
                mode=mode,
            )
        except Exception as e:
            st.error(f"Generation failed: {e}")
            generated = None

        if generated is not None:
            # Show settings used
            settings_text = (
                f"Mode={mode} | Order={effective_order} | Length={len(generated.split()) if mode=='word' else len(generated)} | Seed={seed_to_use if seed_enabled else 'random'}"
            )
            st.caption(settings_text)

            out_col, act_col = st.columns([3, 1])
            with out_col:
                st.markdown("**Generated output**")
                st.code(generated, language="text")
            with act_col:
                st.markdown("**Actions**")
                st.download_button(
                    label="Download .txt",
                    data=generated,
                    file_name="markov_output.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
                if st.button("Copy to clipboard", use_container_width=True, key="copy_btn"):
                    st.info("Highlight the code block above and use Ctrl/Cmd+C.")

            # Save locally only on explicit request
            save_checked = st.checkbox("Save output to outputs/ (local)", value=False)
            if save_checked:
                out_path = OUTPUT_DIR / "last_output.txt"
                out_path.write_text(generated, encoding="utf-8")
                st.success(f"Saved to {out_path.name}")

    # --- Explanation -----------------------------------------------------
    with st.expander("📖 How Markov-chain transitions work"):
        st.write(
            "A Markov model predicts the next token from the preceding `n` tokens. "
            "Counts are converted to frequency-weighted sampling — more common next tokens appear more often, but any observed transition is possible. "
            "Higher orders create more locally coherent sequences, yet small corpora may leave gaps, causing the generator to restart or back off. "
            "The algorithm is purely statistical; it does not understand meaning, grammar, or context."
        )
        st.markdown("- Word-level: tokens are whitespace-separated words, punctuation stays attached.")
        st.markdown("- Character-level: each character is a token.")
        st.markdown("- Reproducibility: same corpus + settings + seed gives identical text.")

    st.divider()
    st.caption("Prodigy InfoTech Generative AI Internship Task 03 — Markov Text Studio. Built with Python, Streamlit, and a real statistical Markov chain.")
    st.info("No neural models or external APIs are used. All generation is performed locally by counting transition frequencies.")


if __name__ == "__main__":
    main()
