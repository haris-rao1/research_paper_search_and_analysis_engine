# ============================================
# clusters.py - Document Clustering
# ============================================
# This file groups similar documents together into "clusters".\n#
# Think of it like:\n# - You have 1000 papers scattered everywhere
# - Clustering automatically organizes them into groups (e.g., 20 clusters)
# - Each cluster contains papers about similar topics
# - Cluster 1 might be \"Neural Networks\", Cluster 2 might be \"Quantum Computing\", etc.
#
# We use HDBSCAN algorithm - it's smart because:
# 1. Automatically decides how many clusters to make
# 2. Can handle noise (papers that don't fit anywhere)
# 3. Works well with high-dimensional data (like embeddings)
#
# Endpoints:
#  POST /clusters/build - Create clusters from embeddings
#  GET  /clusters - List all clusters
#  GET  /clusters/{id}/docs - See papers in a cluster
#  GET  /clusters/umap - Get visualization coordinates

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict, Any, List
from collections import defaultdict
import numpy as np
import warnings
from pymongo import UpdateOne

# Suppress sklearn deprecation warnings from hdbscan
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'", category=FutureWarning)

from db import db, papers_collection

try:
    import hdbscan
    _HAS_HDBSCAN = True
except Exception:
    hdbscan = None
    _HAS_HDBSCAN = False

try:
    from sklearn.cluster import KMeans
    _HAS_SKLEARN = True
except Exception:
    KMeans = None
    _HAS_SKLEARN = False

try:
    import umap
    _HAS_UMAP = True
except Exception:
    umap = None
    _HAS_UMAP = False

router = APIRouter()
clusters_collection = db["clusters_meta"]
embeddings_collection = db["embeddings"]  # you already have this collection

def _load_embeddings_and_doc_ids():
    """
    Load embeddings from embeddings_collection (fallback to papers.embedding if needed).
    Returns (doc_ids_list, emb_matrix) or (None, None) if some documents missing embeddings.
    """
    # Try embeddings_collection first (it stores per-doc embeddings)
    cursor = embeddings_collection.find({}, {"doc_id": 1, "embedding": 1}).sort("doc_id", 1)
    emb_list = []
    doc_ids = []
    for d in cursor:
        vec = d.get("embedding")
        if vec is None:
            return None, None
        doc_ids.append(int(d["doc_id"]))
        emb_list.append(np.array(vec, dtype="float32"))
    if emb_list:
        emb_matrix = np.vstack(emb_list)
        # L2-normalize rows
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = emb_matrix / norms
        return doc_ids, emb_matrix

    # Fallback: try reading 'embedding' from papers_collection
    cursor = papers_collection.find({}, {"doc_id": 1, "embedding": 1}).sort("doc_id", 1)
    emb_list = []
    doc_ids = []
    for d in cursor:
        vec = d.get("embedding")
        if vec is None:
            return None, None
        doc_ids.append(int(d["doc_id"]))
        emb_list.append(np.array(vec, dtype="float32"))
    if emb_list:
        emb_matrix = np.vstack(emb_list)
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_matrix = emb_matrix / norms
        return doc_ids, emb_matrix

    return None, None


@router.post("/clusters/build")
def build_clusters(
    method: str = Query("hdbscan", description="hdbscan or kmeans"),
    min_cluster_size: int = Query(8, description="HDBSCAN min_cluster_size (or n_clusters for KMeans)"),
    min_samples: Optional[int] = Query(None, description="HDBSCAN min_samples (defaults to min_cluster_size//2)"),
    use_umap_pre_reduce: bool = Query(True, description="Run UMAP to 10 dims before clustering"),
    umap_n_neighbors: int = Query(15),
    umap_min_dist: float = Query(0.1)
) -> Dict[str, Any]:
    """
    Build clusters from stored embeddings. Updates papers collection with 'cluster_id' and 'cluster_score'.
    Stores clusters metadata in clusters_meta collection.
    """
    print("Loading embeddings for clustering...") 
    doc_ids, emb_matrix = _load_embeddings_and_doc_ids()
    if emb_matrix is None:
        raise HTTPException(status_code=400, detail="Embeddings missing for some or all documents. Run embeddings build first.")

    X = emb_matrix
    # Optional pre-reduction to 10 dims to denoise
    if use_umap_pre_reduce and _HAS_UMAP:
        reducer = umap.UMAP(n_components=10, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, random_state=42)
        X_reduced = reducer.fit_transform(X)
    else:
        X_reduced = X

    labels = None
    probabilities = None

    if method.lower() == "hdbscan" and _HAS_HDBSCAN:
        min_samples_val = int(min_samples) if min_samples is not None else max(1, min_cluster_size // 2)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=min_samples_val, prediction_data=True)
        labels = clusterer.fit_predict(X_reduced)
        # Try to get soft membership/probabilities
        try:
            # hdbscan can provide soft membership vectors, but fallback gracefully
            probs = hdbscan.all_points_membership_vectors(clusterer)
            # if probs is matrix, pick for each point the value for its assigned label when available
            if isinstance(probs, np.ndarray):
                probabilities = []
                for i, lab in enumerate(labels):
                    if lab < 0:
                        probabilities.append(0.0)
                    else:
                        if lab < probs.shape[1]:
                            probabilities.append(float(probs[i, lab]))
                        else:
                            probabilities.append(float(np.max(probs[i, :])))
                probabilities = np.array(probabilities, dtype="float32")
            else:
                # fallback to clusterer.probabilities_ if present (per-sample)
                p_attr = getattr(clusterer, "probabilities_", None)
                if p_attr is not None:
                    probabilities = np.array(p_attr, dtype="float32")
                else:
                    probabilities = np.zeros(len(labels), dtype="float32")
        except Exception:
            p_attr = getattr(clusterer, "probabilities_", None)
            probabilities = np.array(p_attr, dtype="float32") if p_attr is not None else np.zeros(len(labels), dtype="float32")

    elif method.lower() == "kmeans":
        if not _HAS_SKLEARN:
            raise HTTPException(status_code=500, detail="scikit-learn is required for kmeans")
        # interpret min_cluster_size param as n_clusters when using kmeans
        n_clusters = max(2, int(min_cluster_size))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_reduced)
        # simple "probability" measure: inverse normalized distance to centroid
        centroids = kmeans.cluster_centers_
        dists = np.linalg.norm(X_reduced[:, None, :] - centroids[None, :, :], axis=2)
        maxd = dists.max(axis=1, keepdims=True) + 1e-12
        inv = 1.0 - (dists / maxd)
        probabilities = inv[np.arange(inv.shape[0]), labels].astype("float32")
    else:
        # fallback: error if hdbscan requested but not available
        raise HTTPException(status_code=400, detail="Unknown clustering method or required package not installed")

    # Save cluster assignments per doc using bulk write for better performance
    cluster_ids = [int(x) for x in labels.tolist()]
    cluster_scores = [float(x) for x in (probabilities.tolist() if probabilities is not None else [0.0]*len(cluster_ids))]

    # Use bulk_write instead of individual update_one calls for better performance
    bulk_operations = [
        UpdateOne(
            {"doc_id": int(doc_id)}, 
            {"$set": {"cluster_id": int(cid), "cluster_score": float(score)}}
        )
        for doc_id, cid, score in zip(doc_ids, cluster_ids, cluster_scores)
    ]
    
    if bulk_operations:
        papers_collection.bulk_write(bulk_operations, ordered=False)

    # Compute cluster metadata: members, centroid (mean embedding), dominant LDA topic
    cluster_members = defaultdict(list)
    for doc_id, cid in zip(doc_ids, cluster_ids):
        cluster_members[int(cid)].append(int(doc_id))

    # Bulk fetch all embeddings and topics for ALL cluster members at once
    all_member_ids = [did for members in cluster_members.values() for did in members]
    
    # Bulk fetch embeddings
    embeddings_map = {}
    if all_member_ids:
        emb_cursor = embeddings_collection.find({"doc_id": {"$in": all_member_ids}}, {"doc_id": 1, "embedding": 1})
        embeddings_map = {d["doc_id"]: d["embedding"] for d in emb_cursor}
        
        # Fallback: bulk fetch from papers if embeddings missing
        missing_ids = [did for did in all_member_ids if did not in embeddings_map]
        if missing_ids:
            papers_cursor = papers_collection.find({"doc_id": {"$in": missing_ids}}, {"doc_id": 1, "embedding": 1})
            for d in papers_cursor:
                if d.get("embedding"):
                    embeddings_map[d["doc_id"]] = d["embedding"]
    
    # Bulk fetch topics
    topics_map = {}
    if all_member_ids:
        topics_cursor = papers_collection.find({"doc_id": {"$in": all_member_ids}}, {"doc_id": 1, "topic_lda_id": 1})
        topics_map = {d["doc_id"]: d.get("topic_lda_id") for d in topics_cursor if d.get("topic_lda_id") is not None}
    
    clusters_meta = []
    for cid, members in cluster_members.items():
        print(f"Processing cluster {cid} with {len(members)} members...")
        # centroid over original embeddings using bulk-fetched data
        member_embs = []
        for did in members:
            vec = embeddings_map.get(did)
            if vec is not None:
                member_embs.append(np.array(vec, dtype="float32"))
        
        if member_embs:
            M = np.vstack(member_embs)
            c = np.mean(M, axis=0)
            n = np.linalg.norm(c)
            centroid = (c / n).tolist() if n != 0 else c.tolist()
        else:
            centroid = None

        # determine dominant LDA topic among members using bulk-fetched data
        topic_counts = defaultdict(int)
        for did in members:
            topic_id = topics_map.get(did)
            if topic_id is not None:
                topic_counts[int(topic_id)] += 1
        dominant_topic = max(topic_counts.items(), key=lambda x: x[1])[0] if topic_counts else None

        clusters_meta.append({
            "cluster_id": int(cid),
            "size": int(len(members)),
            "centroid": centroid,
            "top_topic": int(dominant_topic) if dominant_topic is not None else None,
            "sample_doc_ids": members[:10]
        })

    clusters_collection.replace_one({"_id": "clusters_meta"}, {"_id": "clusters_meta", "clusters": clusters_meta}, upsert=True)
    print(f"Saved cluster metadata for {len(cluster_members)} clusters.")
    return {"status": "ok", "num_docs": len(doc_ids), "num_clusters": len(cluster_members), "clusters_sample": clusters_meta[:5]}


@router.get("/clusters")
def list_clusters():
    meta = clusters_collection.find_one({"_id": "clusters_meta"})
    if not meta:
        return {"clusters": [], "message": "No clusters built yet"}
    return {"clusters": meta.get("clusters", [])}


@router.get("/clusters/{cluster_id}/docs")
def get_cluster_docs(cluster_id: int, limit: int = Query(50, gt=0, le=1000), skip: int = Query(0, ge=0)):
    cursor = papers_collection.find({"cluster_id": int(cluster_id)}, {"_id":0, "doc_id":1, "title":1, "summary":1, "authors":1, "published":1, "umap_x":1, "umap_y":1}).skip(skip).limit(limit)
    docs = list(cursor)
    return {"cluster_id": cluster_id, "count": len(docs), "docs": docs}


@router.get("/clusters/umap")
def get_clusters_umap(limit: Optional[int] = None):
    proj = papers_collection.find({}, {"_id":0, "doc_id":1, "title":1, "summary":1, "authors":1, "published":1, "umap_x":1, "umap_y":1, "cluster_id":1})
    points = []
    for p in proj:
        if "umap_x" not in p or "umap_y" not in p:
            continue
        points.append({
            "doc_id": p["doc_id"],
            "title": p.get("title"),
            "snippet": (p.get("summary") or "")[:300],
            "authors": p.get("authors"),
            "published": p.get("published"),
            "x": float(p["umap_x"]),
            "y": float(p["umap_y"]),
            "cluster_id": p.get("cluster_id", -1)
        })
        if limit and len(points) >= limit:
            break
    return {"count": len(points), "points": points}