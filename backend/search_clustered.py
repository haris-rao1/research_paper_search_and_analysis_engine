# search_clustered.py
# Advanced search endpoint implementing cluster-first -> topic -> restricted scoring.
#
# GET /search/clustered?q=...&method=semantic|bm25|tfidf|cosine|hybrid&k=10&cluster_top_m=1&cluster_threshold=0.45
#
from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any, Optional, Set
import numpy as np

from db import db, papers_collection, inverted_index_collection, embeddings_collection
from bm25 import bm25_score_for_query
from tfIdf import tfidf_score_for_query
from cosineSimilarity import cosine_similarity_for_query
from embeddings import get_model, l2_normalize_rows
from preprocessor import preprocessing

router = APIRouter()
clusters_collection = db["clusters_meta"]
topics_collection = db["topics_lda"]

def _load_cluster_centroids():
    meta = clusters_collection.find_one({"_id": "clusters_meta"})
    if not meta:
        return [], np.zeros((0,0), dtype="float32")
    clusters = meta.get("clusters", [])
    ids = []
    centroids = []
    for c in clusters:
        centroid = c.get("centroid")
        if centroid is None:
            continue
        ids.append(int(c["cluster_id"]))
        centroids.append(np.array(centroid, dtype="float32"))
    if centroids:
        M = np.vstack(centroids)
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        M = M / norms
        return ids, M
    return [], np.zeros((0,0), dtype="float32")

def _load_full_inverted_index():
    doc = inverted_index_collection.find_one({"_id": "inverted_index"})
    if not doc:
        return {}, {}, 0, 0.0
    inv = {}
    for item in doc.get("index", []):
        term = item[0]
        df = item[1]
        postings_list = item[2]
        postings = {int(p[0]): p[1] for p in postings_list}
        inv[term] = {"df": df, "postings": postings}
    meta = doc.get("meta", {})
    doc_lengths = {int(k): int(v) for k, v in meta.get("doc_lengths", {}).items()} if meta.get("doc_lengths") else {}
    total_docs = int(meta.get("total_docs", len(doc_lengths)))
    avgdl = float(meta.get("avg_doc_length", sum(doc_lengths.values()) / total_docs if total_docs else 0.0))
    return inv, doc_lengths, total_docs, avgdl

def _build_restricted_index(full_inv: Dict[str, Dict], candidate_set: Set[int]):
    restricted = {}
    for term, data in full_inv.items():
        postings = data.get("postings", {})
        filtered = {doc_id: tf for doc_id, tf in postings.items() if doc_id in candidate_set}
        if filtered:
            restricted[term] = {"df": data.get("df", 0), "postings": filtered}
    return restricted

@router.get("/search/clustered")
def search_clustered(
    q: str,
    method: str = Query("semantic"),
    k: int = Query(10, gt=0, le=200),
    cluster_mode: bool = Query(True),
    cluster_top_m: int = Query(1, description="number of top clusters to consider"),
    cluster_threshold: float = Query(0.45, description="min cosine to accept cluster; if not met fallback global")
) -> Dict[str, Any]:
    if not q or not q.strip():
        return {"query": q, "results": []}

    selected_clusters = []
    cluster_scores = []
    selected_cluster_meta = None
    candidate_doc_ids: List[int] = []

    full_inv, doc_lengths, total_docs, avgdl = _load_full_inverted_index()

    # 1) find cluster(s) via embedding centroid similarity
    if cluster_mode:
        ids, centroids = _load_cluster_centroids()
        if len(ids) == 0:
            cluster_mode = False
        else:
            model = get_model()
            q_emb = model.encode([q], convert_to_numpy=True).astype("float32")
            q_emb = l2_normalize_rows(q_emb)[0]
            if centroids.shape[0] > 0:
                sims = centroids @ q_emb
                top_idx = np.argsort(-sims)[:cluster_top_m]
                for idx in top_idx:
                    score = float(sims[idx])
                    cid = ids[int(idx)]
                    if score >= cluster_threshold or cluster_top_m > 1:
                        selected_clusters.append(int(cid))
                        cluster_scores.append(float(score))
                if selected_clusters:
                    meta = clusters_collection.find_one({"_id": "clusters_meta"})
                    if meta:
                        for c in meta.get("clusters", []):
                            if c["cluster_id"] == selected_clusters[0]:
                                selected_cluster_meta = c
                                break

    # 2) candidate set = docs in selected clusters (or all docs if cluster selection disabled)
    if cluster_mode and selected_clusters:
        cand_set = set()
        for cid in selected_clusters:
            for d in papers_collection.find({"cluster_id": int(cid)}, {"doc_id":1}):
                cand_set.add(int(d["doc_id"]))
        candidate_doc_ids = list(cand_set)
    else:
        candidate_doc_ids = [int(d["doc_id"]) for d in papers_collection.find({}, {"doc_id":1})]

    if not candidate_doc_ids:
        return {"query": q, "results": [], "selected_cluster_meta": selected_cluster_meta}

    candidate_set = set(candidate_doc_ids)
    results = []

    # 3) run requested ranking restricted to candidate set
    if method in ("bm25", "tfidf", "cosine", "hybrid"):
        processed = preprocessing({0: q})
        q_terms = processed.get(0, [])
        if not q_terms:
            return {"query": q, "results": []}

        restricted_inv = _build_restricted_index(full_inv, candidate_set)

        if method == "bm25":
            ranked = bm25_score_for_query(q_terms, restricted_inv, doc_lengths, total_docs, avgdl, top_k=k)
        elif method == "tfidf":
            ranked = tfidf_score_for_query(q_terms, restricted_inv, doc_lengths, total_docs, top_k=k)
        elif method == "cosine":
            ranked = cosine_similarity_for_query(q_terms, restricted_inv, doc_lengths, total_docs, top_k=k)
        else:  # hybrid
            r_b = bm25_score_for_query(q_terms, restricted_inv, doc_lengths, total_docs, avgdl, top_k=k)
            r_t = tfidf_score_for_query(q_terms, restricted_inv, doc_lengths, total_docs, top_k=k)
            r_c = cosine_similarity_for_query(q_terms, restricted_inv, doc_lengths, total_docs, top_k=k)
            bm = {d: s for d, s in r_b}
            tf = {d: s for d, s in r_t}
            cs = {d: s for d, s in r_c}
            max_b = max(bm.values()) if bm else 1.0
            max_t = max(tf.values()) if tf else 1.0
            max_c = max(cs.values()) if cs else 1.0
            doc_ids_set = set(list(bm.keys()) + list(tf.keys()) + list(cs.keys()))
            combined = {}
            for d in doc_ids_set:
                sb = (bm.get(d, 0.0) / max_b) if max_b else 0.0
                st = (tf.get(d, 0.0) / max_t) if max_t else 0.0
                sc = (cs.get(d, 0.0) / max_c) if max_c else 0.0
                combined[d] = (sb + st + sc) / 3.0
            ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]

        # format results
        for doc_id, score in ranked:
            doc = papers_collection.find_one({"doc_id": int(doc_id)}, {"_id":0, "doc_id":1, "title":1, "summary":1, "authors":1, "published":1, "link":1, "cluster_id":1, "topic_lda_id":1})
            if not doc:
                continue
            results.append({
                "doc_id": doc["doc_id"],
                "title": doc.get("title"),
                "snippet": (doc.get("summary") or "")[:300],
                "score": float(score),
                "authors": doc.get("authors"),
                "published": doc.get("published"),
                "link": doc.get("link"),
                "cluster_id": doc.get("cluster_id"),
                "topic_lda_id": doc.get("topic_lda_id")
            })

    elif method == "semantic":
        model = get_model()
        q_emb = model.encode([q], convert_to_numpy=True).astype("float32")
        q_emb = l2_normalize_rows(q_emb)[0]
        
        # Bulk fetch embeddings for all candidates
        embeddings_map = {}
        emb_cursor = embeddings_collection.find({"doc_id": {"$in": candidate_doc_ids}}, {"doc_id": 1, "embedding": 1})
        embeddings_map = {d["doc_id"]: d["embedding"] for d in emb_cursor}
        
        # Fallback: bulk fetch from papers if embeddings missing
        missing_ids = [did for did in candidate_doc_ids if did not in embeddings_map]
        if missing_ids:
            papers_cursor = papers_collection.find({"doc_id": {"$in": missing_ids}}, {"doc_id": 1, "embedding": 1})
            for d in papers_cursor:
                if d.get("embedding"):
                    embeddings_map[d["doc_id"]] = d["embedding"]
        
        # Build embedding matrix
        emb_list = []
        id_order = []
        for did in candidate_doc_ids:
            vec = embeddings_map.get(did)
            if vec is None:
                continue
            emb_list.append(np.array(vec, dtype="float32"))
            id_order.append(int(did))
        
        if not emb_list:
            return {"query": q, "results": []}
        
        emb_matrix = np.vstack(emb_list)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = emb_matrix / norms
        scores = (emb_matrix @ q_emb)
        top_idx = np.argsort(-scores)[:k]
        
        # Bulk fetch document metadata for top results
        top_doc_ids = [id_order[int(idx)] for idx in top_idx]
        docs_cursor = papers_collection.find(
            {"doc_id": {"$in": top_doc_ids}}, 
            {"_id":0, "doc_id":1, "title":1, "summary":1, "authors":1, "published":1, "link":1, "cluster_id":1, "topic_lda_id":1}
        )
        docs_map = {d["doc_id"]: d for d in docs_cursor}
        
        for idx in top_idx:
            did = id_order[int(idx)]
            sc = float(scores[idx])
            doc = docs_map.get(did)
            if not doc:
                continue
            results.append({
                "doc_id": doc["doc_id"],
                "title": doc.get("title"),
                "snippet": (doc.get("summary") or "")[:300],
                "score": sc,
                "authors": doc.get("authors"),
                "published": doc.get("published"),
                "link": doc.get("link"),
                "cluster_id": doc.get("cluster_id"),
                "topic_lda_id": doc.get("topic_lda_id")
            })
    else:
        raise HTTPException(status_code=400, detail="Unknown method")

    # 4) enforce cluster restriction: keep only docs in candidate_set when cluster_mode active
    filtered_before = len(results)
    if cluster_mode and selected_clusters:
        results = [r for r in results if r.get("doc_id") in candidate_set]
    filtered_after = len(results)
    filtered_out = filtered_before - filtered_after

    # 5) map predicted topic from cluster meta if available
    predicted_topic = None
    if selected_cluster_meta and selected_cluster_meta.get("top_topic") is not None:
        tid = int(selected_cluster_meta["top_topic"])
        tmeta = topics_collection.find_one({"_id": "lda_meta"})
        if tmeta:
            for t in tmeta.get("topics", []):
                if t.get("topic_id") == tid:
                    predicted_topic = {"topic_id": tid, "top_terms": t.get("top_terms", [])}
                    break

    return {
        "query": q,
        "method": method,
        "k": k,
        "selected_cluster_ids": selected_clusters,
        "cluster_scores": cluster_scores,
        "selected_cluster_meta": selected_cluster_meta,
        "predicted_topic": predicted_topic,
        "filtered_out_count": filtered_out,
        "results": results
    }