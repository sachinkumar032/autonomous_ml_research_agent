"""
retriever.py
Lightweight local RAG using TF-IDF + cosine similarity -- no external
embedding model download needed, matching the blueprint's advice to
"begin with a simpler local store" (Section 18 suggests ChromaDB; this
is a simpler substitute that needs zero setup and no network call).

The corpus combines two things per blueprint Section 17:
1. Static ML guidance docs (knowledge_base.py)
2. This project's own experiment reports (pulled live from experiment
   history, so retrieval reflects what's actually been tried)
"""

import sys
sys.path.append("src/ml")
sys.path.append("src/rag")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_base import GUIDANCE_DOCS
from experiment_manager import load_all_experiment_logs


def _experiment_logs_as_docs() -> list[dict]:
    """Turns each logged experiment into a short retrievable text document."""
    docs = []
    for log in load_all_experiment_logs():
        text = (
            f"Experiment #{log.get('experiment_id')}: {log.get('name')} "
            f"({log.get('model_family')}) with params {log.get('params')}. "
            f"Result: F1={log.get('metrics', {}).get('f1')}, "
            f"accuracy={log.get('metrics', {}).get('accuracy')}, "
            f"ROC-AUC={log.get('metrics', {}).get('roc_auc')}."
        )
        docs.append({
            "id": f"experiment_{log.get('experiment_id')}",
            "title": f"Experiment {log.get('experiment_id')}: {log.get('name')}",
            "text": text,
            "source": "experiment_history",
        })
    return docs


def build_corpus() -> list[dict]:
    """Combines static guidance docs with live experiment history."""
    guidance = [{**d, "source": "ml_guidance"} for d in GUIDANCE_DOCS]
    return guidance + _experiment_logs_as_docs()


def retrieve(query: str, top_k: int = 3, corpus: list[dict] = None) -> list[dict]:
    """
    Returns the top_k most relevant documents for the query, each with
    a similarity score. Rebuilds the corpus fresh each call (cheap at
    this scale, and guarantees experiment history is always current).
    """
    if corpus is None:
        corpus = build_corpus()

    if not corpus:
        return []

    texts = [f"{d['title']}. {d['text']}" for d in corpus]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts + [query])

    query_vec = matrix[-1]
    doc_vecs = matrix[:-1]
    scores = cosine_similarity(query_vec, doc_vecs).flatten()

    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)

    results = []
    for doc, score in ranked[:top_k]:
        if score <= 0:
            continue
        results.append({
            "id": doc["id"],
            "title": doc["title"],
            "text": doc["text"],
            "source": doc["source"],
            "relevance_score": round(float(score), 3),
        })
    return results
