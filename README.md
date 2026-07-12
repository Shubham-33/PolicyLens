# PolicyLens 🔍

**A compliance-grade answer engine for policy documents — self-hostable, so it
never leaves the bank's own infrastructure.** Upload one or more policy documents,
ask a question in plain English, and get an answer **grounded only in your
documents** — with clickable citations that jump to the exact source passage,
NotebookLM-style. If the answer isn't in your documents, PolicyLens says so
instead of guessing. Every answer is traceable to its source: an audit trail, not
just a chat reply.

Built for the IDBI / Hack2Skill hackathon.

[![CI](https://github.com/Shubham-33/PolicyLens/actions/workflows/ci.yml/badge.svg)](https://github.com/Shubham-33/PolicyLens/actions/workflows/ci.yml)
![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)

---

## The problem

Bank policy and product documents are long, dense, and revised constantly. The
same handful of questions ("What's the minimum balance? What's the ATM charge?
When does an account go dormant?") gets asked thousands of times.

- **Volume** — a single retail branch fields *dozens* of policy questions a day;
  a contact centre, thousands. Most are re-asks of the same few facts.
- **Cost of a wrong answer** — quoting the wrong charge or eligibility rule is a
  **mis-selling / compliance exposure**, not a typo. In banking, a confidently
  wrong answer is worse than no answer.
- **Why not a generic chatbot** — a public LLM can't ingest confidential policy
  docs (data-residency), and it *hallucinates*. Neither is acceptable here.

PolicyLens answers from the bank's own documents, on the bank's own infra, and
**cites its source every time** — so the answer is auditable.

## The approach: grounding, not fine-tuning

PolicyLens uses **Retrieval-Augmented Generation (RAG)** with hybrid retrieval:

1. **Ingest** — each document is split into overlapping passages and (when a key
   is present) embedded with NVIDIA `nv-embedqa-e5-v5`.
2. **Retrieve** — a **hybrid** ranker blends semantic similarity (embeddings) with
   lexical TF-IDF, so a question about an *"inactive"* account still finds the
   passage that says *"dormant"*. Falls back to pure TF-IDF with no key.
3. **Generate** — NVIDIA NIM (`llama-3.1-8b-instruct`) answers using *only* the
   retrieved passages and cites each one; the UI links every `[n]` to its source.

Why not fine-tune a model on the FAQ? Fine-tuning bakes facts into weights you
can't cite, goes stale the moment a policy changes, and can still hallucinate.
RAG keeps every answer **traceable to a source** and updates the instant you
upload a new document — the right call for compliance-sensitive banking content.
(Fine-tuning a domain adapter remains a roadmap option for tone/terminology.)

## Key features

- 📎 **Citations you can click** — every claim links to the exact source passage.
- 🧠 **Hybrid semantic + keyword retrieval** — finds answers by *meaning*, not just
  matching words; recovers paraphrases keyword search misses.
- 📚 **Multi-document** — load several policies; ask across all of them at once;
  each citation names its source document.
- 🚫 **No hallucination** — off-topic questions are refused, not answered.
- 🔒 **Private per session** — each browser session has its own isolated corpus;
  concurrent users never see each other's documents.
- 🏦 **Self-hostable** — open model via NVIDIA NIM; no data leaves your infra.
- ⚡ **Degrades gracefully** — with no API key it falls back to deterministic
  lexical retrieval + extractive answers, so the demo (and the whole test suite)
  never depends on the network.
- ♿ **Accessible** — semantic HTML, ARIA, keyboard shortcuts, WCAG-AA contrast
  (Lighthouse: Accessibility 100, Best Practices 100).

## Architecture

```
web/
├── app.py            Flask routes + middleware (gzip, caching, security headers)
├── rag.py            Retrieval core — TF-IDF + hybrid semantic ranking (pure Python)
├── embed.py          NVIDIA embeddings for semantic recall, with a no-op fallback
├── nim.py            NVIDIA NIM grounded generation + offline extractive fallback
├── ingest.py         PDF / txt / md  ->  page texts
├── session_store.py  disk-backed per-session corpora (multi-worker safe)
├── sample_data.py    bundled sample policy for a cold-click demo
├── templates/        single-page UI
├── static/           app.js (citation wiring) + app.css
└── tests/            96 tests, 100% coverage gate
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
