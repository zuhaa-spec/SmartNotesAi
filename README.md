# Smart Notes AI

An AI-themed **study assistant** for college students: paste lecture notes, get an **extractive summary**, and practice with **auto-generated multiple-choice quizzes**. Built for quick demos (e.g. GDG Technical Team interviews) using **Python + Streamlit** only—no paid APIs, API keys, or large model downloads.

## Features

- **Notes input** — large text area, optional `.txt` upload, one-click sample notes, validation for empty/short text
- **Smart summarization** — extractive summary via word frequencies and sentence scoring (original wording preserved)
- **Quiz generation** — rule-based fill-in-the-blank MCQs with shuffled options and distractors from your notes
- **Interactive quiz** — one question at a time, Previous/Next, submit, score, review with explanations
- **Results dashboard** — metrics, performance summary, simple bar chart, restart quiz

## Technologies

| Tool | Role |
|------|------|
| Python 3.10+ | Logic, text processing |
| Streamlit | Web UI |
| `re`, `collections.Counter`, `random` | Tokenization, scoring, quiz options |

## Installation

```bash
cd smart-notes-ai
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## How summarization works

1. **Sentence split** — notes are split on `.`, `!`, `?` into sentences (minimum length filter).
2. **Word frequencies** — words are lowercased; common **stopwords** and very short tokens are ignored; a frequency table is built over all sentences.
3. **Sentence score** — for each sentence, sum the frequencies of its content words and normalize by √word count so long sentences are not unfairly favored.
4. **Selection** — pick the top *k* sentences by score (*k* depends on short/medium/long), then **re-sort by original order** so the summary still reads naturally.
5. **Output** — selected sentences are concatenated unchanged (**extractive**, not generative).

## How quiz questions are generated

1. Sentences are ranked by the same importance score used in summarization.
2. For each sentence, **candidate keywords** (length ≥ 4, not stopwords, frequent in the document) are chosen.
3. The first suitable keyword is replaced with `______` to form a fill-in-the-blank stem.
4. **One correct option** (the original word) plus **distractors** from other frequent words in the notes (with fallbacks if the pool is small) make four options.
5. Options are **shuffled**; the correct index is stored for grading.
6. Unsuitable sentences (no keyword, duplicate options, too few distractors) are **skipped**.

## Current limitations

- Summaries can feel repetitive or miss nuance compared to abstractive/neural summarizers.
- Quiz quality depends on note style; lists, tables, or very short bullets produce fewer questions.
- Distractors are plausible by word frequency, not semantic similarity.
- English-oriented tokenization; mixed languages may behave poorly.
- No user accounts or saved history (session-only).

## Future improvements

- **True NLP/ML**: sentence embeddings (e.g. Sentence-BERT) for better sentence ranking; abstractive summarization (BART, T5, or small local LLMs).
- **Smarter QG**: dedicated question-generation models or LLM prompts with citation to source sentences.
- **Export** PDF/Markdown summary and quiz; spaced-repetition deck export (Anki).
- **Accessibility** and mobile layout polish.

## Project structure

```
smart-notes-ai/
    app.py              # UI + summarization + quiz logic
    requirements.txt
    README.md
```

## License

MIT — use freely for learning and interview demos.
