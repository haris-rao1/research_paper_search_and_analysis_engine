import math
from typing import Dict, List, Tuple


def bm25_score_for_query(
    query_terms: List[str],
    inverted_index: Dict[str, Dict],
    doc_lengths: Dict[int, int],
    total_docs: int,
    avg_doc_length: float,
    k1: float = 1.5,
    b: float = 0.75,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Compute BM25 scores for the given preprocessed query_terms.

    inverted_index: { term: {"df": int, "postings": {doc_id: tf, ...}}, ... }
    doc_lengths: { doc_id: length, ... }
    total_docs: total number of documents in corpus (N)
    avg_doc_length: average document length (avgdl)
    """
    if not query_terms:
        return []

    if total_docs <= 0 or avg_doc_length <= 0:
        return []

    scores: Dict[int, float] = {}

    for term in query_terms:
        entry = inverted_index.get(term)
        if not entry:
            continue

        df = entry.get("df", 0)
        postings = entry.get("postings", {})

        # BM25 IDF with shown smoothing
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

        for doc_id, tf in postings.items():
            dl = doc_lengths.get(doc_id, 0)
            if dl == 0:
                continue

            length_norm = 1 - b + b * (dl / avg_doc_length)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * length_norm
            term_score = idf * (numerator / denominator)
            scores[doc_id] = scores.get(doc_id, 0.0) + term_score

    # Return top_k sorted by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]