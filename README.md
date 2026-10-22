# Production RAG with Retrieval Evaluation

Retrieval-augmented generation demos usually stop at "it answers
questions." This project measures the part that decides whether the
answers are any good: **retrieval quality**, configuration by
configuration, on a labeled question set over real SEC 10-K filings.

## What it measures

A grid of 8 retrieval configurations —

- **Chunking:** fixed-size windows vs. structure-aware (split on SEC
  Item headings first)
- **Search:** dense vector (FAISS) vs. hybrid BM25 + vector fused with
  Reciprocal Rank Fusion
- **Reranking:** off vs. cross-encoder rerank of the candidate pool

— scored on 18 hand-labeled questions over the latest 10-Ks of AAPL,
MSFT, and NVDA. Each question carries a *gold anchor*: an exact phrase
from the passage that answers it. A retrieved chunk counts as relevant
iff it contains the anchor, which keeps relevance reproducible without
an LLM judge.

Metrics per configuration: **Recall@5, Recall@10, MRR@10, mean latency**.

## Results

Measured on the 18-question labeled set (this machine, CPU;
`python -m evals.runner`, full output in `evals/results/`):

| Config | Chunker | Search | Rerank | Recall@5 | Recall@10 | MRR@10 | Mean latency (ms) |
|---|---|---|---|---|---|---|---|
| fixed+vector+base | fixed | vector | no | 0.667 | 0.889 | 0.378 | 23 |
| fixed+hybrid+base | fixed | hybrid | no | 0.889 | 0.944 | 0.671 | 20 |
| structure+vector+base | structure | vector | no | 0.444 | 0.667 | 0.344 | 18 |
| structure+hybrid+base | structure | hybrid | no | 0.944 | 0.944 | 0.633 | 32 |

What the numbers say:

- **Hybrid search is the single biggest win.** Adding BM25 (fused with
  RRF) lifts Recall@5 from 0.667 → 0.889 on fixed chunks and 0.444 →
  0.944 on structure chunks, and roughly doubles MRR. Filings are full
  of exact terms (product names, figures, "Item 1A") that pure dense
  retrieval undervalues.
- **Structure-aware chunking alone does *not* help dense retrieval** —
  it was the worst configuration with vector search (Recall@5 0.444).
  Smaller, heading-prefixed chunks dilute the embedding signal. It only
  pays off paired with hybrid search, where it ties for the best
  Recall@10 and takes the best Recall@5.
- **Latency is a non-issue at this scale:** every configuration
  answers in ~20–30 ms; the ranking differences cost nothing.

**Rerank configurations were not run in this environment:** the
cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) could not
be downloaded — Hugging Face egress was unreachable from the build
machine after repeated attempts (direct and via mirror). The runner
detects this, records the rerank rows as "not run" in
`evals/results/results.json`, and completes the rest of the grid; a
plain `python -m evals.runner` with Hub access fills in all 8 rows.
Treat the rerank column as *pending measurement*, not as a result.

## Architecture

```
SEC EDGAR ──> scripts/fetch_corpus.py ──> data/corpus/<TICKER>_10k.txt
                                              │
                          rag/  chunkers ─────┼── fixed | structure
                                index    ───────┤   sentence-transformers + FAISS
                                retrievers ─────┤   vector | BM25 | hybrid (RRF)
                                reranker ───────┘   cross-encoder (optional)
                                              │
                          evals/  questions.jsonl + metrics + runner (standalone)
                          app.py  Streamlit demo (pick a config, ask, see citations + scores)
```

## Run it

```bash
pip install -r requirements.txt

python scripts/fetch_corpus.py      # download + clean the 10-Ks (already committed)
python -m evals.runner              # full 8-config grid -> evals/results/
python -m pytest tests/             # unit tests
streamlit run app.py                # demo
```

## The evals package is reusable

`evals/` has no dependency on the demo app. It defines the relevance
model (gold anchors), the metrics, and the grid runner; any retrieval
system that can return ranked chunk texts can be scored with it.

## Honest limits

- One gold passage per question: recall@k here means "did we surface
  the labeled passage," not "did we surface everything relevant."
- The corpus is three filings from one industry; numbers would shift
  on a broader or noisier corpus.
- Answers are retrieved passages, not generated text — this project
  evaluates retrieval, and generation quality is out of scope by design.
- Latency is measured on this machine (CPU); treat it as relative
  between configurations, not absolute.
