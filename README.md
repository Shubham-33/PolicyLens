# PolicyLens 🔍

**A retrieval-grounded policy & FAQ assistant.** Upload a policy document, ask a
question in plain English, and get an answer that is **grounded only in your
document** — with clickable citations that jump to the exact source passage,
NotebookLM-style. If the answer isn't in your documents, PolicyLens says so
instead of guessing.

Built for the IDBI / Hack2Skill hackathon.

[![CI](https://github.com/Shubham-33/PolicyLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Shubham-33/PolicyLens/actions/workflows/ci.yml)
![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

---

## The problem

Bank policy documents are long, dense, and change often. Customers and staff ask
the same questions ("What's the minimum balance?", "What's the ATM charge?") and
either dig through PDFs or get an answer with **no traceable source**. A plain
chatbot will confidently *hallucinate* a wrong charge — unacceptable for banking.

## The approach: grounding, not fine-tuning

PolicyLens uses **Retrieval-Augmented Generation (RAG)**:

1. **Ingest** — the document is split into overlapping passages.
2. **Retrieve** — a TF-IDF index finds the passages most relevant to the question.
3. **Generate** — NVIDIA NIM (`llama-3.1-8b-instruct`) answers using *only* those
   passages and cites each one; the UI links every `[n]` marker to its source.

Why not fine-tune a model on the FAQ? Fine-tuning bakes facts into weights you
can't cite, goes stale the moment a policy changes, and can still hallucinate.
RAG keeps every answer **traceable to a source** and updates the instant you
upload a new document — the right call for compliance-sensitive banking content.
(Fine-tuning a domain adapter remains a roadmap option for tone/terminology.)

## Key features

- 📎 **Citations you can click** — every claim links to the exact source passage.
- 🚫 **No hallucination** — off-topic questions are refused, not answered.
- 📄 **Bring your own doc** — drag-and-drop PDF, `.txt`, or `.md`, or load the sample.
- ⚡ **Works offline** — with no API key it falls back to a deterministic extractive
  answer, so the demo (and the whole test suite) never depends on the network.
- ♿ **Accessible** — semantic HTML, ARIA, keyboard shortcuts, WCAG-AA contrast.

## Architecture

```
web/
├── app.py          Flask routes + middleware (gzip, caching, security headers)
├── rag.py          TF-IDF retrieval — pure Python, deterministic, zero ML deps
├── nim.py          NVIDIA NIM grounded generation + offline extractive fallback
├── ingest.py       PDF / txt / md  ->  page texts
├── sample_data.py  bundled sample policy for a cold-click demo
├── templates/      single-page UI
├── static/         app.js (citation wiring) + app.css
└── tests/          64 tests, 100% coverage gate
```

## Run locally

```bash
cd web
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Optional: enable live NVIDIA NIM answers (works without it too)
export NIM_API_KEY="nvapi-..."      # from https://build.nvidia.com

python app.py                        # http://localhost:5050
```

Without a key the badge reads **offline mode** and answers are extractive; with a
key it reads **NVIDIA NIM · live** and answers are fully generated.

## Test, lint

```bash
cd web
pytest            # 64 tests, fails under 100% coverage
ruff check .      # lint
```

## Deploy

Any host that runs a `Procfile` works (Cloud Run, Render, Railway). The app reads
`PORT` from the environment and binds `0.0.0.0`. Set `NIM_API_KEY` as a secret.
See [`deploy.sh`](deploy.sh) for a one-shot Google Cloud Run deploy.

## Tech

Flask · vanilla JS · Tailwind (CDN) · NVIDIA NIM (`llama-3.1-8b-instruct`) ·
pure-Python TF-IDF · pytest + coverage · ruff.

## Note

The bundled sample policy is illustrative and is **not** an official IDBI Bank
document. Upload your own policy to get real answers.
