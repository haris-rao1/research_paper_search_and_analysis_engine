import math
from typing import Dict, List, Tuple


def tfidf_score_for_query(
    query_terms: List[str],
    inverted_index: Dict[str, Dict],
    doc_lengths: Dict[int, int],
    total_docs: int,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Compute TF-IDF scores (document-centric) for the query.

    TF normalized by document length. IDF = log(N / (df + 1)) + 1
    """
    if not query_terms:
        return []

    if total_docs <= 0:
        return []

    scores: Dict[int, float] = {}

    for term in query_terms:
        entry = inverted_index.get(term)
        if not entry:
            continue

        df = entry.get("df", 0)
        postings = entry.get("postings", {})

        idf = math.log(total_docs / (df + 1)) + 1.0

        for doc_id, tf in postings.items():
            dl = doc_lengths.get(doc_id, 0)
            if dl == 0:
                continue
            tf_normalized = tf / dl
            scores[doc_id] = scores.get(doc_id, 0.0) + tf_normalized * idf

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]