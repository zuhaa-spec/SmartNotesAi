"""
Smart Notes AI — AI-powered study assistant (rule-based extractive summary + quiz).
Built for GDG Technical Team interview demo. No paid APIs or heavy ML models required.
"""

import random
import re
from collections import Counter
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Constants & sample data
# ---------------------------------------------------------------------------

MIN_NOTES_CHARS = 80
MIN_SENTENCES_FOR_QUIZ = 3
STOPWORDS = frozenset(
    """
    a an the and or but if then else when at by for with about against between
    into through during before after above below to from up down in out on off
    over under again further once here there all each few more most other some
    such no nor not only own same so than too very can will just don should now
    is are was were be been being have has had do does did doing would could
    might must shall may it its this that these those i you he she we they what
    which who whom how why where of as also than
    """.split()
)

SAMPLE_NOTES = """
Photosynthesis is the process by which green plants convert light energy into chemical energy.
Chlorophyll, found in chloroplasts, absorbs sunlight most strongly in the blue and red wavelengths.
The light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH.
The Calvin cycle takes place in the stroma and uses CO2 to build glucose.
Stomata are small pores on leaves that allow gas exchange for photosynthesis.
Factors affecting the rate of photosynthesis include light intensity, carbon dioxide concentration, and temperature.
Cellular respiration breaks down glucose to release ATP for the cell's work.
Mitochondria are often called the powerhouse of the cell because they produce most of the cell's ATP.
The equation for photosynthesis is often summarized as: 6CO2 + 6H2O + light → C6H12O6 + 6O2.
Understanding these concepts is essential for biology exams on plant metabolism.
""".strip()


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple punctuation rules."""
    text = normalize_whitespace(text)
    if not text:
        return []
    # Split on . ! ? followed by space or end; keep reasonable chunks
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = []
    for p in parts:
        p = p.strip()
        if len(p) >= 20 and re.search(r"[a-zA-Z]", p):
            sentences.append(p)
    return sentences


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def word_frequencies(sentences: list[str]) -> Counter:
    words = []
    for s in sentences:
        for w in tokenize_words(s):
            if w not in STOPWORDS and len(w) > 2:
                words.append(w)
    return Counter(words)


def score_sentence(sentence: str, freqs: Counter) -> float:
    """Higher score = more important words from the document."""
    words = [w for w in tokenize_words(sentence) if w not in STOPWORDS and len(w) > 2]
    if not words:
        return 0.0
    return sum(freqs.get(w, 0) for w in words) / (len(words) ** 0.5)


# ---------------------------------------------------------------------------
# Summarization (extractive)
# ---------------------------------------------------------------------------


def extractive_summary(notes: str, length: str = "medium") -> tuple[str, list[str]]:
    """
    Select top-scoring sentences in original order (extractive, not generative).
    length: short (3), medium (5), long (8) — capped by available sentences.
    """
    sentences = split_sentences(notes)
    if not sentences:
        return "", []

    caps = {"short": 3, "medium": 5, "long": 8}
    k = min(caps.get(length, 5), len(sentences))

    freqs = word_frequencies(sentences)
    scored = [(i, s, score_sentence(s, freqs)) for i, s in enumerate(sentences)]
    # Pick top k by score, then sort by original index to preserve reading flow
    top = sorted(sorted(scored, key=lambda x: -x[2])[:k], key=lambda x: x[0])
    selected = [s for _, s, _ in top]
    summary_text = " ".join(selected)
    return summary_text, selected


# ---------------------------------------------------------------------------
# Quiz generation (rule-based fill-in-the-blank MCQ)
# ---------------------------------------------------------------------------


def _candidate_keywords(sentence: str, freqs: Counter) -> list[str]:
    """Important words in the sentence for blanking."""
    words = tokenize_words(sentence)
    seen = set()
    candidates = []
    for w in words:
        wl = w.lower()
        if wl in STOPWORDS or len(wl) < 4:
            continue
        if wl in seen:
            continue
        seen.add(wl)
        # Prefer content words with decent doc frequency
        score = freqs.get(wl, 1)
        candidates.append((wl, score, w))
    candidates.sort(key=lambda x: -x[1])
    return [orig for _, _, orig in candidates[:5]]


def _distractors(correct: str, all_words: Counter, n: int = 3) -> list[str]:
    """Pick plausible wrong answers from other frequent words in notes."""
    correct_l = correct.lower()
    pool = [w for w, _ in all_words.most_common(40) if w != correct_l and len(w) > 3]
    random.shuffle(pool)
    distractors = []
    for w in pool:
        if w.lower() == correct_l:
            continue
        # Title-case if original was capitalized
        display = w.capitalize() if correct[0].isupper() else w
        if display not in distractors and display.lower() != correct_l:
            distractors.append(display)
        if len(distractors) >= n - 1:
            break
    # Fallback generic biology/study terms if pool too small
    fallbacks = ["energy", "oxygen", "proteins", "nucleus", "membrane", "enzymes", "water"]
    for fb in fallbacks:
        if len(distractors) >= n - 1:
            break
        d = fb.capitalize() if correct[0].isupper() else fb
        if d.lower() != correct_l and d not in distractors:
            distractors.append(d)
    return distractors[: n - 1]


def generate_questions(notes: str, max_questions: int = 8) -> list[dict]:
    """
    Build MCQs by replacing one keyword with ______ in factual sentences.
    Each dict: question, options, correct_index, explanation, source_sentence
    """
    sentences = split_sentences(notes)
    if len(sentences) < MIN_SENTENCES_FOR_QUIZ:
        return []

    freqs = word_frequencies(sentences)
    all_word_counter = Counter()
    for s in sentences:
        all_word_counter.update(
            w for w in tokenize_words(s) if w not in STOPWORDS and len(w) > 3
        )

    questions: list[dict] = []
    used_sentences: set[str] = set()

    # Prefer higher-scoring sentences for questions
    ranked = sorted(
        enumerate(sentences),
        key=lambda x: -score_sentence(x[1], freqs),
    )

    for _, sentence in ranked:
        if len(questions) >= max_questions:
            break
        if sentence in used_sentences:
            continue

        keywords = _candidate_keywords(sentence, freqs)
        if not keywords:
            continue

        blank_word = None
        question_text = None
        for kw in keywords:
            # Case-insensitive replace first occurrence only
            pattern = re.compile(re.escape(kw), re.IGNORECASE)
            if not pattern.search(sentence):
                continue
            question_text = pattern.sub("______", sentence, count=1)
            blank_word = kw
            break

        if not question_text or not blank_word:
            continue

        # Skip if blank leaves trivial question
        if question_text.count("______") != 1:
            continue

        correct = blank_word
        wrongs = _distractors(correct, all_word_counter, n=3)
        if len(wrongs) < 2:
            continue

        options = [correct] + wrongs
        # Ensure unique options (case-insensitive)
        unique_opts: list[str] = []
        seen_lower: set[str] = set()
        for o in options:
            ol = o.lower()
            if ol not in seen_lower:
                seen_lower.add(ol)
                unique_opts.append(o)
        if len(unique_opts) < 3:
            continue
        while len(unique_opts) < 4:
            extra = _distractors(correct, all_word_counter, n=5)
            for e in extra:
                if e.lower() not in seen_lower:
                    unique_opts.append(e)
                    seen_lower.add(e.lower())
                if len(unique_opts) >= 4:
                    break
            break
        if len(unique_opts) < 4:
            continue

        options = unique_opts[:4]
        random.shuffle(options)
        correct_index = next(i for i, o in enumerate(options) if o.lower() == correct.lower())

        questions.append(
            {
                "question": f"Fill in the blank: {question_text}",
                "options": options,
                "correct_index": correct_index,
                "explanation": f"The missing term is **{correct}**, from your notes: \"{sentence}\"",
                "source_sentence": sentence,
            }
        )
        used_sentences.add(sentence)

    return questions


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compute_score(questions: list[dict], user_answers: list[Optional[int]]) -> dict:
    total = len(questions)
    if total == 0:
        return {"correct": 0, "total": 0, "percent": 0.0}

    correct = 0
    for i, q in enumerate(questions):
        if i < len(user_answers) and user_answers[i] is not None:
            if user_answers[i] == q["correct_index"]:
                correct += 1

    percent = round(100.0 * correct / total, 1)
    return {"correct": correct, "total": total, "percent": percent}


def performance_summary(percent: float) -> str:
    if percent >= 90:
        return "Excellent — you know this material well. Review any missed blanks briefly."
    if percent >= 70:
        return "Good job — solid understanding with a few gaps to revisit in your notes."
    if percent >= 50:
        return "Fair — re-read the summary and focus on the sentences you missed."
    return "Keep studying — use the summary and explanations to reinforce key terms."


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        /* Warm off-white dashboard */
        .stApp {
            background-color: #F7F5F0;
            color: #2C2C2C;
        }
        [data-testid="stSidebar"] {
            background-color: #EFEBE3;
            border-right: 1px solid #D8D2C8;
        }
        h1, h2, h3 {
            color: #2C2C2C !important;
            font-weight: 600 !important;
        }
        .sn-card {
            background: #FFFFFF;
            border: 1px solid #E2DDD4;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(44, 44, 44, 0.06);
        }
        .sn-badge {
            display: inline-block;
            background: #E8F0EA;
            color: #3D5C45;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .sn-muted {
            color: #5C5C5C;
            font-size: 0.95rem;
        }
        div.stButton > button[kind="primary"] {
            background-color: #4A7C59;
            border-color: #4A7C59;
            color: white;
            border-radius: 8px;
            font-weight: 500;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #3D6A4A;
            border-color: #3D6A4A;
        }
        div[data-testid="stMetricValue"] {
            color: #3D5C45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "notes_text": "",
        "summary_text": "",
        "summary_sentences": [],
        "questions": [],
        "quiz_index": 0,
        "user_answers": [],
        "quiz_submitted": False,
        "processed": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_notes_from_inputs() -> str:
    return st.session_state.get("notes_text", "").strip()


def process_notes(notes: str, summary_length: str, num_questions: int) -> None:
    with st.spinner("Analyzing your notes (word frequencies & sentence scoring)…"):
        summary, sents = extractive_summary(notes, summary_length)
        qs = generate_questions(notes, max_questions=num_questions)
    st.session_state.summary_text = summary
    st.session_state.summary_sentences = sents
    st.session_state.questions = qs
    st.session_state.user_answers = [None] * len(qs)
    st.session_state.quiz_index = 0
    st.session_state.quiz_submitted = False
    st.session_state.processed = True


def page_notes_input() -> None:
    st.markdown('<span class="sn-badge">Step 1</span>', unsafe_allow_html=True)
    st.subheader("Your study notes")

    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("Load sample notes", use_container_width=True):
            st.session_state.notes_text = SAMPLE_NOTES
            st.rerun()

    notes = st.text_area(
        "Paste notes here",
        height=220,
        placeholder="Paste lecture notes, textbook excerpts, or revision material…",
        key="notes_text",
        label_visibility="collapsed",
    )

    uploaded = st.file_uploader("Optional: upload a .txt file", type=["txt"])
    if uploaded is not None:
        try:
            content = uploaded.read().decode("utf-8", errors="replace")
            st.session_state.notes_text = content
            st.success(f"Loaded **{uploaded.name}** into the editor.")
        except Exception as e:
            st.error(f"Could not read file: {e}")

    notes = get_notes_from_inputs()
    if not notes:
        st.info("Add notes above or load the sample to continue.")
        return
    if len(notes) < MIN_NOTES_CHARS:
        st.warning(
            f"Notes look too short (minimum ~{MIN_NOTES_CHARS} characters). "
            "Add more sentences so summarization and quiz generation work well."
        )
        return

    st.markdown("---")
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        summary_length = st.select_slider(
            "Summary length",
            options=["short", "medium", "long"],
            value="medium",
        )
    with c2:
        num_questions = st.slider("Max quiz questions", 3, 10, 6)
    with c3:
        st.write("")
        st.write("")
        if st.button("Generate summary & quiz", type="primary", use_container_width=True):
            process_notes(notes, summary_length, num_questions)
            st.rerun()


def page_summary() -> None:
    st.subheader("Smart summary")
    st.markdown(
        '<p class="sn-muted">Extractive summarization — we score sentences by important word '
        "frequencies and pick the top ones in original order. Wording is unchanged; "
        "this is not a neural language model.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.processed:
        st.info("Process your notes from the **Home** page first.")
        return

    if not st.session_state.summary_text:
        st.warning("Could not build a summary. Try longer notes with full sentences.")
        return

    st.markdown('<div class="sn-card">', unsafe_allow_html=True)
    st.markdown(st.session_state.summary_text)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Sentences included in the summary"):
        for i, s in enumerate(st.session_state.summary_sentences, 1):
            st.markdown(f"{i}. {s}")


def page_quiz() -> None:
    st.subheader("Practice quiz")
    st.markdown(
        '<p class="sn-muted">Rule-based generator — key terms are turned into fill-in-the-blank '
        "multiple choice. Quality depends on your notes; unsuitable sentences are skipped.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.processed:
        st.info("Process your notes from the **Home** page first.")
        return

    questions = st.session_state.questions
    if not questions:
        st.warning(
            "Not enough suitable sentences for quiz questions. "
            "Try adding more factual sentences (definitions, processes, key terms)."
        )
        return

    if st.session_state.quiz_submitted:
        render_results(questions)
        return

    idx = st.session_state.quiz_index
    q = questions[idx]
    total = len(questions)

    st.progress((idx + 1) / total, text=f"Question {idx + 1} of {total}")

    st.markdown('<div class="sn-card">', unsafe_allow_html=True)
    st.markdown(f"**{q['question']}**")
    st.markdown("</div>", unsafe_allow_html=True)

    # Stable key per question index for radio
    current = st.session_state.user_answers[idx]
    choice = st.radio(
        "Choose an answer",
        options=list(range(len(q["options"]))),
        format_func=lambda i: q["options"][i],
        index=current,
        key=f"quiz_radio_{idx}",
        label_visibility="collapsed",
    )
    if choice is not None:
        st.session_state.user_answers[idx] = choice

    nav1, nav2, nav3 = st.columns([1, 1, 2])
    with nav1:
        if st.button("Previous", disabled=idx == 0, use_container_width=True):
            st.session_state.quiz_index = max(0, idx - 1)
            st.rerun()
    with nav2:
        if idx < total - 1:
            if st.button("Next", use_container_width=True):
                st.session_state.quiz_index = idx + 1
                st.rerun()
        else:
            if st.button("Submit quiz", type="primary", use_container_width=True):
                st.session_state.quiz_submitted = True
                st.rerun()


def render_results(questions: list[dict]) -> None:
    stats = compute_score(questions, st.session_state.user_answers)

    st.markdown("### Results dashboard")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Score", f"{stats['correct']} / {stats['total']}")
    m2.metric("Percentage", f"{stats['percent']}%")
    m3.metric("Correct", stats["correct"])
    m4.metric("Questions", stats["total"])

    st.markdown(
        f'<div class="sn-card"><strong>Performance summary:</strong> '
        f"{performance_summary(stats['percent'])}</div>",
        unsafe_allow_html=True,
    )

    if stats["total"] > 0:
        incorrect = stats["total"] - stats["correct"]
        st.bar_chart(
            {
                "Correct": {"Count": stats["correct"]},
                "Incorrect": {"Count": incorrect},
            }
        )

    st.markdown("#### Review answers")
    for i, q in enumerate(questions):
        ua = st.session_state.user_answers[i]
        correct_i = q["correct_index"]
        ok = ua == correct_i
        icon = "✅" if ok else "❌"
        st.markdown(f"**{icon} Question {i + 1}**")
        st.markdown(q["question"])
        if ua is not None:
            st.markdown(f"Your answer: *{q['options'][ua]}*")
        st.markdown(f"Correct answer: *{q['options'][correct_i]}*")
        st.markdown(q["explanation"])
        st.markdown("---")

    if st.button("Restart quiz", type="primary"):
        st.session_state.quiz_index = 0
        st.session_state.user_answers = [None] * len(questions)
        st.session_state.quiz_submitted = False
        st.rerun()


def page_about() -> None:
    st.subheader("About Smart Notes AI")
    st.markdown(
        """
        **Smart Notes AI** helps college students revise faster by:

        - **Summarizing** notes with an explainable extractive algorithm (word frequency + sentence scores).
        - **Generating** multiple-choice practice from your own material using simple linguistic rules.
        - **Quizzing** you interactively with scored feedback tied back to your notes.

        This demo intentionally avoids paid APIs and large model downloads so it runs anywhere in minutes.

        **Honest labeling:** Features here are *rule-based text processing*, not trained deep learning.
        Future versions could plug in summarization or QG models (e.g. BERT, T5, or API-based LLMs).
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Smart Notes AI",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    init_session_state()

    st.title("Smart Notes AI")
    st.caption("Study assistant · extractive summaries · rule-based quizzes")

    with st.sidebar:
        st.markdown("### Navigate")
        page = st.radio(
            "Section",
            ["Home", "Summary", "Quiz", "About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            '<p class="sn-muted" style="font-size:0.85rem;">Muted green accents · '
            "student-friendly dashboard</p>",
            unsafe_allow_html=True,
        )

    if page == "Home":
        page_notes_input()
    elif page == "Summary":
        page_summary()
    elif page == "Quiz":
        page_quiz()
    else:
        page_about()


if __name__ == "__main__":
    main()
