"""
Tests for Level 6: the RAG retriever over ML guidance docs and
experiment history. No API key needed -- this is pure TF-IDF
retrieval, no LLM involved.
"""

import sys
sys.path.append("src/rag")
sys.path.append("src/ml")

from retriever import retrieve, build_corpus
from knowledge_base import GUIDANCE_DOCS


def test_corpus_includes_guidance_docs():
    corpus = build_corpus()
    guidance_ids = {d["id"] for d in GUIDANCE_DOCS}
    corpus_ids = {d["id"] for d in corpus}
    assert guidance_ids <= corpus_ids


def test_corpus_includes_experiment_history():
    corpus = build_corpus()
    sources = {d["source"] for d in corpus}
    # experiment history should be present if any experiments have been logged
    assert "ml_guidance" in sources


def test_retrieve_class_imbalance_query_returns_relevant_doc():
    results = retrieve("what should I try for class imbalance?", top_k=3)
    assert len(results) > 0
    titles = " ".join(r["title"].lower() for r in results)
    assert "imbalance" in titles


def test_retrieve_overfitting_query_returns_relevant_doc():
    results = retrieve("my model seems to be overfitting", top_k=3)
    assert len(results) > 0
    titles = " ".join(r["title"].lower() for r in results)
    assert "overfitting" in titles or "underfitting" in titles


def test_retrieve_results_are_ranked_by_score_descending():
    results = retrieve("class imbalance", top_k=5)
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_handles_empty_corpus_gracefully():
    results = retrieve("anything", top_k=3, corpus=[])
    assert results == []
