# search_enhanced.py
# FastAPI router implementing "cluster-first -> topic" query flow and candidate-restricted ranking.
#
# Endpoint:
# - GET /search/advanced?q=...&method=semantic|bm25|tfidf|cosine|hybrid&k=10&cluster_mode=true
#
# It expects existing functions for BM25/TF-IDF/cosine ranking present in your repo:
# - bm25.bm25_score_for_query(...)
# - tfidf.tfidf_score_for_query(...)
# - cosine_similarity.cosine_similarity_for_query(...)
#
# And expects embeddings available in papers.embedding and clusters metadata in clusters_meta.
#
from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any, Optional
import numpy as np

from db import db, papers_collection
from embeddings import get_model, l2_normalize_rows  # if you have embeddings.py with get_model; else import SentenceTransformer
from bm25 import bm25_score_for_query
from tfidf import tfidf_score_for_query
from cosine_similarity import cosine_similarity_for_query

clusters_collection = db["clusters_meta"]
topics_collection = db["topics_lda"]

router = APIRouter()

def _load_cluster_centroids():
    meta = clusters_collection.find_one({"_id": "clusters_meta"})
    if not meta:
        return [], []
    clusters = meta.get("clusters", [])
    centroids = []
    ids = []
    for c in clusters:
        if c.get("centroid") is None:
            continue
        centroids.append(np.array(c["centroid"], dtype="float32"))
        ids.append(int(c["cluster_id"]))
    if centroids:
        M = np.vstack(centroids)
        return ids, l2_normalize_rows(M)
    else:
        return [], np.zeros((0,0), dtype="float32")

@router.get("/search/advanced")
def search_advanced(q: str,
                    method: str = Query("semantic"),
                    k: int = Query(10, gt=0, le=200),
                    cluster_mode: bool = Query(True, description="If true, find best cluster first and search inside it"),
                    cluster_top_m: int = Query(1, description="How many top clusters to consider"),
                    cluster_threshold: float = Query(0.45, description="Minimum cosine to accept cluster; else fallback global")
                    ) -> Dict[str, Any]:
    """
    Advanced search: cluster-first pipeline.
    - Compute query embedding, match to clusters via centroid cosine.
    - If cluster accepted, restrict candidate docs to cluster members.
    - Then run requested method on candidate set.
    """
    if not q or not q.strip():
        return {"query": q, "results": []}

    # If cluster_mode is enabled, find best cluster(s)
    selected_cluster_ids = []
    cluster_scores = []
    selected_cluster_meta = None
    if cluster_mode:
        ids, centroids = _load_cluster_centroids()
        if len(ids) == 0:
            # no clusters built -> fallback to global search
            cluster_mode = False
        else:
            model = get_model()
            q_emb = model.encode([q], convert_to_numpy=True).astype("float32")
            q_emb = l2_normalize_rows(q_emb)[0]
            # centroids is shape (num_clusters, dim)
            if centroids.shape[0] > 0:
                sims = centroids @ q_emb  # cosine scores
                top_idx = np.argsort(-sims)[:cluster_top_m]
                for idx in top_idx:
                    score = float(sims[idx])
                    cid = ids[int(idx)]
                    if score >= cluster_threshold or cluster_top_m > 1:
                        selected_cluster_ids.append(int(cid))
                        cluster_scores.append(float(score))
                if selected_cluster_ids:
                    # pick the top cluster meta for reporting
                    meta = clusters_collection.find_one({"_id": "clusters_meta"})
                    if meta:
                        clusters_meta = meta.get("clusters", [])
                        for c in clusters_meta:
                            if c["cluster_id"] == selected_cluster_ids[0]:
                                selected_cluster_meta = c
                                break
            else:
                cluster_mode = False

    # Determine candidate doc_ids
    candidate_doc_ids = []
    if cluster_mode and len(selected_cluster_ids) > 0:
        # union docs from selected clusters
        for cid in selected_cluster_ids:
            cursor = papers_collection.find({"cluster_id": int(cid)}, {"doc_id": 1})
            for d in cursor:
                candidate_doc_ids.append(int(d["doc_id"]))
    else:
        # global search => all docs
        cursor = papers_collection.find({}, {"doc_id":1})
        candidate_doc_ids = [int(d["doc_id"]) for d in cursor]

    if not candidate_doc_ids:
        return {"query": q, "results": [], "selected_cluster": None}

    # Now run the requested scoring method, but restricted to candidate_doc_ids
    results = []
    if method == "bm25":
        # call bm25 with restriction - assume function allows candidate set param, otherwise we'll compute scores then filter
        try:
            ranked = bm25_score_for_query(q, candidate_doc_ids=candidate_doc_ids, top_k=k)
        except TypeError:
            # fallback: run bm25 over full collection but filter (less efficient)
            ranked_all = bm25_score_for_query(q, top_k=len(candidate_doc_ids)*2)
            ranked = [r for r in ranked_all if r["doc_id"] in set(candidate_doc_ids)][:k]
        results = ranked
    elif method == "tfidf":
        try:
            ranked = tfidf_score_for_query(q, candidate_doc_ids=candidate_doc_ids, top_k=k)
        except TypeError:
            ranked_all = tfidf_score_for_query(q, top_k=len(candidate_doc_ids)*2)
            ranked = [r for r in ranked_all if r["doc_id"] in set(candidate_doc_ids)][:k]
        results = ranked
    elif method == "cosine":
        try:
            ranked = cosine_similarity_for_query(q, candidate_doc_ids=candidate_doc_ids, top_k=k)
        except TypeError:
            ranked_all = cosine_similarity_for_query(q, top_k=len(candidate_doc_ids)*2)
            ranked = [r for r in ranked_all if r["doc_id"] in set(candidate_doc_ids)][:k]
        results = ranked
    elif method == "semantic":
        # compute query embedding and brute-force dot with candidate embeddings
        model = get_model()
        q_emb = model.encode([q], convert_to_numpy=True).astype("float32")
        q_emb = l2_normalize_rows(q_emb)[0]
        # load candidate embeddings
        emb_list = []
        id_order = []
        for did in candidate_doc_ids:
            doc = papers_collection.find_one({"doc_id": int(did)}, {"embedding":1, "title":1, "summary":1, "authors":1, "published":1, "link":1})
            if not doc or not doc.get("embedding"):
                continue
            emb_list.append(np.array(doc["embedding"], dtype="float32"))
            id_order.append(int(did))
        if not emb_list:
            return {"query": q, "results": []}
        emb_matrix = np.vstack(emb_list)
        # normalize
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = emb_matrix / norms
        scores = (emb_matrix @ q_emb)
        top_idx = np.argsort(-scores)[:k]
        results = []
        for idx in top_idx:
            did = id_order[int(idx)]
            sc = float(scores[idx])
            doc = papers_collection.find_one({"doc_id": int(did)}, {"_id":0, "doc_id":1, "title":1, "summary":1, "authors":1, "published":1, "link":1, "cluster_id":1, "topic_lda_id":1})
            if not doc:
                continue
            results.append({
                "doc_id": doc["doc_id"],
                "title": doc.get("title"),
                "snippet": (doc.get("summary") or "")[:300],
                "score": sc,
                "published": doc.get("published"),
                "authors": doc.get("authors"),
                "link": doc.get("link"),
                "cluster_id": doc.get("cluster_id"),
                "topic_lda_id": doc.get("topic_lda_id")
            })
    elif method == "hybrid":
        # simple hybrid: run semantic to get top N, then re-rank with BM25 on that small set
        # do semantic shortlist of 200 candidates
        model = get_model()
        q_emb = model.encode([q], convert_to_numpy=True).astype("float32")
        q_emb = l2_normalize_rows(q_emb)[0]
        emb_list = []
        id_order = []
        for did in candidate_doc_ids:
            doc = papers_collection.find_one({"doc_id": int(did)}, {"embedding":1})
            if not doc or not doc.get("embedding"):
                continue
            emb_list.append(np.array(doc["embedding"], dtype="float32"))
            id_order.append(int(did))
        if not emb_list:
            return {"query": q, "results": []}
        emb_matrix = np.vstack(emb_list)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = emb_matrix / norms
        scores = (emb_matrix @ q_emb)
        topN = 200
        top_idx = np.argsort(-scores)[:min(topN, len(scores))]
        candidate_subset = [id_order[int(i)] for i in top_idx]
        # now run bm25 but ideally with candidate restriction
        try:
            bm25_ranked = bm25_score_for_query(q, candidate_doc_ids=candidate_subset, top_k=k)
            results = bm25_ranked
        except TypeError:
            # fallback: compute bm25 for candidates manually using existing bm25 scorer over full set then filter
            bm25_all = bm25_score_for_query(q, top_k=len(candidate_subset)*2)
            results = [r for r in bm25_all if r["doc_id"] in set(candidate_subset)][:k]
    else:
        raise HTTPException(status_code=400, detail="Unknown method")

    return {
        "query": q,
        "method": method,
        "k": k,
        "selected_cluster_ids": selected_cluster_ids,
        "cluster_scores": cluster_scores,
        "selected_cluster_meta": selected_cluster_meta,
        "results": results
    }