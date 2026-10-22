"""Run the retrieval-configuration grid over the labeled question set.

Grid: 2 chunkers x 2 search modes x rerank on/off = 8 configurations.
For each: recall@5, recall@10, MRR@10, mean query latency.

Questions are filtered to their own filing (a question about AAPL is
only ever asked against the AAPL chunks), mirroring how the demo app
works and keeping per-question retrieval honest.

Usage:  python -m evals.runner [--corpus data/corpus] [--out evals/results]
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from rag.chunkers import CHUNKERS
from rag.index import EmbeddingIndex
from rag.pipeline import Config, RetrievalPipeline
from rag.reranker import CrossEncoderReranker

from .metrics import recall_at_k, reciprocal_rank

TOP_K = 10


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    docs = {}
    for path in sorted(corpus_dir.glob("*_10k.txt")):
        ticker = path.name.split("_")[0]
        docs[ticker] = path.read_text(encoding="utf-8")
    if not docs:
        raise RuntimeError(f"no corpus files in {corpus_dir} — run scripts/fetch_corpus.py")
    return docs


def run_grid(documents: dict[str, str], questions: list[dict]) -> list[dict]:
    # Pre-build one embedding index per (chunker, ticker); every config
    # with that chunker shares it. Chunking is deterministic, so pipeline
    # chunk positions line up with the shared index.
    indexes: dict[tuple[str, str], EmbeddingIndex] = {}
    for chunker_name in ("fixed", "structure"):
        for ticker, text in documents.items():
            indexes[(chunker_name, ticker)] = EmbeddingIndex(
                CHUNKERS[chunker_name](text, ticker)
            )
            print(f"indexed {ticker} with {chunker_name}: "
                  f"{len(indexes[(chunker_name, ticker)].chunks)} chunks")

    results = []
    reranker = None  # lazy: only load if a config needs it
    rerank_unavailable = None  # reason string if the model cannot load
    for chunker_name, search_name, use_rerank in itertools.product(
        ["fixed", "structure"], ["vector", "hybrid"], [False, True]
    ):
        config = Config(chunker=chunker_name, search=search_name, rerank=use_rerank)
        if use_rerank:
            if rerank_unavailable is not None:
                results.append(
                    {
                        "config": config.name,
                        "chunker": chunker_name,
                        "search": search_name,
                        "rerank": True,
                        "status": f"not run: {rerank_unavailable}",
                    }
                )
                continue
            if reranker is None:
                try:
                    reranker = CrossEncoderReranker()
                except Exception as exc:  # model download blocked, etc.
                    rerank_unavailable = f"reranker model unavailable ({exc.__class__.__name__})"
                    print(f"WARNING: {rerank_unavailable} — rerank configs will be skipped")
                    results.append(
                        {
                            "config": config.name,
                            "chunker": chunker_name,
                            "search": search_name,
                            "rerank": True,
                            "status": f"not run: {rerank_unavailable}",
                        }
                    )
                    continue
        per_ticker = {
            ticker: RetrievalPipeline(
                {ticker: text},
                config,
                index=indexes[(chunker_name, ticker)],
                reranker=reranker,
            )
            for ticker, text in documents.items()
        }

        r5, r10, mrr, lat = [], [], [], []
        for q in questions:
            pipe = per_ticker[q["ticker"]]
            chunks, latency_ms = pipe.query(q["question"], top_k=TOP_K)
            texts = [c.text for c in chunks]
            r5.append(recall_at_k(texts, q["gold_anchor"], 5))
            r10.append(recall_at_k(texts, q["gold_anchor"], 10))
            mrr.append(reciprocal_rank(texts, q["gold_anchor"], 10))
            lat.append(latency_ms)

        row = {
            "config": config.name,
            "chunker": chunker_name,
            "search": search_name,
            "rerank": use_rerank,
            "n_questions": len(questions),
            "recall@5": round(statistics.mean(r5), 4),
            "recall@10": round(statistics.mean(r10), 4),
            "mrr@10": round(statistics.mean(mrr), 4),
            "mean_latency_ms": round(statistics.mean(lat), 1),
        }
        results.append(row)
        print(
            f"{row['config']:<28} R@5={row['recall@5']:.3f} "
            f"R@10={row['recall@10']:.3f} MRR={row['mrr@10']:.3f} "
            f"lat={row['mean_latency_ms']:.0f}ms"
        )
    return results


def results_markdown(results: list[dict]) -> str:
    lines = [
        "| Config | Chunker | Search | Rerank | Recall@5 | Recall@10 | MRR@10 | Mean latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    notes = []
    for r in results:
        if "status" in r:
            notes.append(f"- {r['config']}: {r['status']}")
            continue
        lines.append(
            f"| {r['config']} | {r['chunker']} | {r['search']} | "
            f"{'yes' if r['rerank'] else 'no'} | {r['recall@5']:.3f} | "
            f"{r['recall@10']:.3f} | {r['mrr@10']:.3f} | {r['mean_latency_ms']:.0f} |"
        )
    if notes:
        lines += ["", "**Not run:**", *notes]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/corpus")
    parser.add_argument("--questions", default="evals/questions.jsonl")
    parser.add_argument("--out", default="evals/results")
    args = parser.parse_args()

    documents = load_corpus(Path(args.corpus))
    questions = load_questions(Path(args.questions))
    print(f"corpus: { {t: len(d.split()) for t, d in documents.items()} }")
    print(f"questions: {len(questions)}")

    results = run_grid(documents, questions)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "results.md").write_text(results_markdown(results))
    print(f"\nwrote {out_dir}/results.json and results.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
