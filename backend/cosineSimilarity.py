import math
from typing import Dict, List, Tuple, Optional


def cosine_similarity_for_query(
    query_terms: List[str],
    inverted_index: Dict[str, Dict],
    doc_lengths: Dict[int, int],
    total_docs: int,
    doc_norms: Optional[Dict[int, float]] = None,
    top_k: int = 10,
) -> List[Tuple[int, float]]:
    """
    Efficient cosine similarity between query and documents using TF-IDF weights.
    - query_terms: list of preprocessed tokens
    - inverted_index: {term: {"df": int, "postings": {doc_id: tf, ...}}, ...}
    - doc_lengths: {doc_id: length, ...} (used to compute tf normalization if doc_norms not provided)
    - total_docs: N
    - doc_norms: optional precomputed L2 norms of document TF-IDF vectors {doc_id: norm}
    - top_k: number of results

    Returns: list of (doc_id, similarity) sorted desc, up to top_k
    """

    if not query_terms or total_docs <= 0:
        return []

    # Build query TF
    qtf: Dict[str, int] = {}
    for t in query_terms:
        qtf[t] = qtf.get(t, 0) + 1
    qlen = len(query_terms)

    # Build query vector (tf-idf). Use IDF = log(N / (df + 1)) + 1 to avoid zero/negatives
    query_vector: Dict[str, float] = {}
    for term, tf in qtf.items():
        entry = inverted_index.get(term)
        if not entry:
            continue
        df = entry.get("df", 0)
        tf_norm = tf / qlen  # normalized query tf
        idf = math.log(total_docs / (df + 1)) + 1.0
        query_vector[term] = tf_norm * idf

    if not query_vector:
        return []

    # Precompute query magnitude
    q_mag = math.sqrt(sum(v * v for v in query_vector.values()))
    if q_mag == 0:
        return []

    # Accumulate dot products by iterating postings of each query term (sparse dot)
    doc_dot: Dict[int, float] = {}
    # If doc_norms not supplied, we will compute doc magnitudes restricted to query terms
    doc_sq_sum: Dict[int, float] = {}

    for term, q_weight in query_vector.items():
        entry = inverted_index.get(term)
        if not entry:
            continue
        postings = entry.get("postings", {})
        # idf already accounted for in q_weight; compute doc-side tf-idf contribution
        df = entry.get("df", 0)
        idf = math.log(total_docs / (df + 1)) + 1.0

        for doc_id, tf in postings.items():
            dl = doc_lengths.get(doc_id, 0)
            if dl == 0:
                continue
            doc_tf_norm = tf / dl  # normalized tf in document
            doc_weight = doc_tf_norm * idf

            doc_dot[doc_id] = doc_dot.get(doc_id, 0.0) + q_weight * doc_weight

            if doc_norms is None:
                # accumulate squared magnitude for terms present in query (partial magnitude)
                doc_sq_sum[doc_id] = doc_sq_sum.get(doc_id, 0.0) + doc_weight * doc_weight

    # Compute final cosine scores
    scores: Dict[int, float] = {}
    for doc_id, dot in doc_dot.items():
        if doc_norms is not None:
            d_mag = doc_norms.get(doc_id, 0.0)
        else:
            # d_mag: sqrt(sum over query-terms (doc_term_weight^2))
            d_mag = math.sqrt(doc_sq_sum.get(doc_id, 0.0))
            # NOTE: this is only partial magnitude if doc has terms outside query.
            # For accurate cosine, precompute full doc norm for all terms.
        if d_mag == 0:
            continue
        scores[doc_id] = dot / (q_mag * d_mag)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]