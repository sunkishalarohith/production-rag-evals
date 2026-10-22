"""Streamlit demo: ask a 10-K a question under any retrieval configuration.

Pick a filing and a configuration (chunker / search / rerank), ask a
question, and see the cited passages the retriever surfaces — alongside
that configuration's measured scores from the eval grid
(evals/results/results.json), so the demo and the measurement are the
same system.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from evals.runner import load_corpus
from rag.chunkers import CHUNKERS
from rag.index import EmbeddingIndex
from rag.pipeline import Config, RetrievalPipeline
from rag.reranker import CrossEncoderReranker

CORPUS_DIR = Path("data/corpus")
RESULTS_PATH = Path("evals/results/results.json")

st.set_page_config(page_title="Production RAG — Retrieval Evaluation", layout="wide")


@st.cache_resource(show_spinner="Loading corpus...")
def corpus() -> dict[str, str]:
    return load_corpus(CORPUS_DIR)


@st.cache_resource(show_spinner="Building index...")
def shared_index(chunker: str, ticker: str) -> EmbeddingIndex:
    return EmbeddingIndex(CHUNKERS[chunker](corpus()[ticker], ticker))


@st.cache_resource(show_spinner="Loading reranker...")
def shared_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


@st.cache_resource(show_spinner="Preparing retriever...")
def pipeline(chunker: str, search: str, rerank: bool, ticker: str) -> RetrievalPipeline:
    config = Config(chunker=chunker, search=search, rerank=rerank)
    return RetrievalPipeline(
        {ticker: corpus()[ticker]},
        config,
        index=shared_index(chunker, ticker),
        reranker=shared_reranker() if rerank else None,
    )


def load_results() -> list[dict] | None:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return None


st.title("Production RAG with Retrieval Evaluation")
st.caption(
    "Questions answered from the latest 10-K filings (SEC EDGAR). "
    "Answers are retrieved passages with citations — and every "
    "configuration is scored on a labeled question set (see evals/)."
)

docs = corpus()
results = load_results()

with st.sidebar:
    st.header("Configuration")
    ticker = st.selectbox("Filing", sorted(docs.keys()))
    chunker = st.radio("Chunking", ["fixed", "structure"], index=1)
    search = st.radio("Search", ["vector", "hybrid"], index=1)
    rerank = st.toggle("Cross-encoder rerank", value=True)

    config_name = f"{chunker}+{search}+{'rerank' if rerank else 'base'}"
    st.divider()
    st.subheader("Measured scores")
    if results:
        row = next((r for r in results if r["config"] == config_name), None)
        if row and "status" in row:
            st.info(f"{config_name}: {row['status']}")
            row = None
        if row:
            st.metric("Recall@5", f"{row['recall@5']:.3f}")
            st.metric("Recall@10", f"{row['recall@10']:.3f}")
            st.metric("MRR@10", f"{row['mrr@10']:.3f}")
            st.metric("Mean latency", f"{row['mean_latency_ms']:.0f} ms")
            st.caption(f"Config: {config_name} · {row['n_questions']} labeled questions")
        with st.expander("All configurations"):
            st.dataframe(
                [
                    {
                        "Config": r["config"],
                        "R@5": r.get("recall@5"),
                        "R@10": r.get("recall@10"),
                        "MRR": r.get("mrr@10"),
                        "ms": r.get("mean_latency_ms"),
                        "Status": r.get("status", "measured"),
                    }
                    for r in results
                ],
                hide_index=True,
            )
    else:
        st.info("Run `python -m evals.runner` to generate measured scores.")

question = st.text_input(
    "Ask a question about the selected filing",
    placeholder="e.g. What risks does the company describe around supply concentration?",
)
ask = st.button("Retrieve passages", type="primary")

if ask and question.strip():
    use_rerank = rerank
    if use_rerank:
        try:
            shared_reranker()
        except Exception:
            st.warning(
                "Reranker model could not be loaded (it downloads on first "
                "use) — showing results without reranking."
            )
            use_rerank = False
    pipe = pipeline(chunker, search, use_rerank, ticker)
    chunks, latency_ms = pipe.query(question, top_k=5)
    st.caption(f"{len(chunks)} passages · retrieval took {latency_ms:.0f} ms (uncached path)")
    for rank, chunk in enumerate(chunks, start=1):
        with st.container(border=True):
            st.markdown(
                f"**#{rank} · {chunk.doc_id} 10-K · {chunk.section or 'preamble'}**"
            )
            st.write(chunk.text[:1200])
elif ask:
    st.warning("Type a question first.")
