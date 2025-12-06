# lda_topics.py
# FastAPI router to train LDA topics from your preprocessed documents,
# store per-document topic assignments, and compute a 2D UMAP projection
# for visualization. Designed to integrate into your existing FastAPI app.
#
# Usage:
#  - POST /topics/lda/build?num_topics=12&passes=10&min_doc_freq=5
#  - GET  /topics/lda
#  - GET  /topics/lda/{topic_id}/docs?limit=20&skip=0
#  - GET  /topics/lda/doc/{doc_id}
#  - GET  /topics/lda/umap?limit=1000
#
# Requirements:
#  pip install gensim umap-learn numpy

from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any, Optional
from collections import defaultdict
import math
import os
import warnings

from db import db, papers_collection  # adapt import if your db module differs
from preprocessor import preprocessing  # reuse your existing preprocessing pipeline

import numpy as np
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel

# UMAP for 2D projection
try:
    warnings.filterwarnings(
        "ignore",
        message="n_jobs value 1 overridden to 1 by setting random_state",
        category=UserWarning,
    )
    import umap
    _HAS_UMAP = True
except Exception:
    umap = None
    _HAS_UMAP = False

router = APIRouter()

# Mongo collections used
topics_collection = db["topics_lda"]
# papers_collection referenced from db.py (must exist in your project)

def _iter_papers():
    """Return list of papers (doc_id, title, summary)."""
    cursor = papers_collection.find({}, {"doc_id": 1, "title": 1, "summary": 1})
    return list(cursor)

@router.post("/topics/lda/build")
def build_lda_topics(
    num_topics: int = Query(12, gt=1),
    passes: int = Query(10, gt=0),
    min_doc_freq: int = Query(5, gt=0),
    max_doc_fraction: float = Query(0.5, gt=0.0, lt=1.0),
    use_embeddings_for_umap: bool = Query(True),
    umap_n_neighbors: int = Query(15),
    umap_min_dist: float = Query(0.1)
) -> Dict[str, Any]:
    """
    Train LDA on your preprocessed documents and store topic metadata & assignments.
    - num_topics: number of topics (K)
    - passes: number of passes for training
    - min_doc_freq: minimum doc frequency to keep a token
    - max_doc_fraction: tokens appearing in > fraction of docs are removed
    - use_embeddings_for_umap: if True and documents have 'embedding' field, use embeddings as UMAP input
    """
    # 1) load papers
    docs = _iter_papers()
    if not docs:
        raise HTTPException(status_code=400, detail="No papers found in papers_collection")

    # Build a mapping index -> text to call your preprocessing pipeline
    texts_map = {}
    id_order = []
    for i, d in enumerate(docs):
        doc_id = int(d["doc_id"])
        id_order.append(doc_id)
        text = ((d.get("title") or "") + "\n" + (d.get("summary") or "")).strip()
        texts_map[i] = text

    # 2) preprocess using your existing pipeline (tokenize, stopwords, lemmatize)
    processed = preprocessing(texts_map)  # returns mapping index -> list(tokens)
    token_lists = [processed.get(i, []) for i in range(len(id_order))]

    # 3) build gensim dictionary & corpus
    dictionary = corpora.Dictionary(token_lists)
    dictionary.filter_extremes(no_below=min_doc_freq, no_above=max_doc_fraction)
    corpus = [dictionary.doc2bow(text) for text in token_lists]

    if not dictionary or len(dictionary.token2id) == 0:
        raise HTTPException(status_code=400, detail="Empty dictionary after filtering - check preprocessing or thresholds")

    # 4) train LDA
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes, random_state=42)

    # 5) extract topics metadata and top words
    topics_meta = []
    for tid in range(num_topics):
        terms = lda.show_topic(tid, topn=12)  # list of (term, weight)
        top_terms = [t for t, _ in terms]
        top_weights = [float(w) for _, w in terms]
        topics_meta.append({"topic_id": int(tid), "top_terms": top_terms, "top_weights": top_weights, "size": 0})

    # 6) per-document topic distributions & primary topic
    doc_topic_assignments = {}
    topic_counts = defaultdict(int)
    for idx, bow in enumerate(corpus):
        doc_topics = lda.get_document_topics(bow, minimum_probability=0.0)  # full distribution
        # convert to list sorted by prob descending
        doc_topics_sorted = sorted([(int(tid), float(prob)) for tid, prob in doc_topics], key=lambda x: -x[1])
        primary_topic = doc_topics_sorted[0][0]
        doc_topic_assignments[id_order[idx]] = {"dist": doc_topics_sorted, "primary": int(primary_topic)}
        topic_counts[primary_topic] += 1

    # update topic sizes
    for t in topics_meta:
        t["size"] = int(topic_counts.get(t["topic_id"], 0))

    # 7) persist topics in topics_collection (replace)
    topics_collection.replace_one({"_id": "lda_meta"}, {"_id": "lda_meta", "num_topics": num_topics, "topics": topics_meta, "dictionary_tokens": len(dictionary.token2id)}, upsert=True)

    # 8) update papers collection with primary topic and topic distribution
    # store topic_lda_id (primary) and topic_lda_dist (list of dicts) and remove older fields if exist
    bulk_ops = []
    for doc_id, info in doc_topic_assignments.items():
        dist_list = [{"topic_id": int(tid), "prob": float(prob)} for tid, prob in info["dist"]]
        papers_collection.update_one({"doc_id": int(doc_id)}, {"$set": {"topic_lda_id": int(info["primary"]), "topic_lda_dist": dist_list}})

    # 9) compute 2D coords for visualization (UMAP)
    coords = []
    umap_input = None
    # try to use embeddings if requested and available
    if use_embeddings_for_umap:
        # check if embeddings present on first documents
        sample = papers_collection.find_one({}, {"embedding": 1})
        if sample and sample.get("embedding"):
            # Bulk fetch embeddings for all docs
            emb_cursor = papers_collection.find(
                {"doc_id": {"$in": [int(d) for d in id_order]}}, 
                {"doc_id": 1, "embedding": 1}
            )
            emb_map = {d["doc_id"]: d.get("embedding") for d in emb_cursor}
            
            emb_list = []
            for doc_id in id_order:
                vec = emb_map.get(int(doc_id))
                if vec is None:
                    emb_list = None
                    break
                emb_list.append(vec)
            if emb_list is not None:
                umap_input = np.array(emb_list, dtype="float32")
    # fallback: use topic distribution vectors (num_topics dims)
    if umap_input is None:
        # build matrix of topic probabilities in same order id_order
        mat = []
        for doc_id in id_order:
            doc = doc_topic_assignments[doc_id]["dist"]
            # create vector length num_topics with probabilities
            vec = np.zeros(num_topics, dtype="float32")
            for tid, prob in doc:
                vec[tid] = prob
            mat.append(vec)
        umap_input = np.vstack(mat)

    # compute UMAP coords if available
    umap_coords = None
    if _HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, random_state=42)
        umap_coords = reducer.fit_transform(umap_input)
    else:
        # if umap not installed, produce simple PCA-like 2D via SVD (fallback)
        try:
            # center
            X = umap_input - np.mean(umap_input, axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            umap_coords = U[:, :2] * S[:2]
        except Exception:
            umap_coords = np.zeros((len(id_order), 2), dtype="float32")

    # store coords back to papers
    for i, doc_id in enumerate(id_order):
        x, y = float(umap_coords[i, 0]), float(umap_coords[i, 1])
        papers_collection.update_one({"doc_id": int(doc_id)}, {"$set": {"umap_x": x, "umap_y": y}})

    return {
        "status": "ok",
        "num_papers": len(id_order),
        "num_topics": num_topics,
        "topics_stored": len(topics_meta),
        "umap_basis": "embeddings" if use_embeddings_for_umap and umap_input is not None else "topic_dist",
        "umap_computed": True
    }

@router.get("/topics/lda")
def list_lda_topics():
    """Return stored topic metadata."""
    meta = topics_collection.find_one({"_id": "lda_meta"})
    if not meta:
        return {"topics": [], "message": "No LDA topics built yet"}
    return {"topics": meta.get("topics", []), "num_topics": meta.get("num_topics", 0)}

@router.get("/topics/lda/{topic_id}/docs")
def get_docs_for_topic(topic_id: int, limit: int = Query(20, gt=0, le=200), skip: int = Query(0, ge=0)):
    """Return documents whose primary LDA topic is topic_id (paginated)."""
    cursor = papers_collection.find({"topic_lda_id": topic_id}, {"_id": 0, "doc_id": 1, "title": 1, "summary": 1, "authors": 1, "published": 1, "umap_x": 1, "umap_y": 1}).skip(skip).limit(limit)
    docs = list(cursor)
    return {"topic_id": topic_id, "count": len(docs), "docs": docs}

@router.get("/topics/lda/doc/{doc_id}")
def get_doc_topic(doc_id: int):
    """Return topic distribution for a specific document (if present)."""
    doc = papers_collection.find_one({"doc_id": int(doc_id)}, {"_id": 0, "doc_id": 1, "title": 1, "topic_lda_id": 1, "topic_lda_dist": 1})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.get("/topics/lda/umap")
def get_umap_points(limit: Optional[int] = None):
    """
    Return umap points for visualization.
    If limit is provided returns up to 'limit' points (useful for sampling).
    """
    proj = papers_collection.find({}, {"_id": 0, "doc_id": 1, "title": 1, "summary": 1, "authors": 1, "published": 1, "umap_x": 1, "umap_y": 1, "topic_lda_id": 1})
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
            "topic_id": p.get("topic_lda_id", -1)
        }) 
        if limit and len(points) >= limit: 
            break
    return {"count": len(points), "points": points}

# To include this router into your main FastAPI app:
# in main.py: from lda_topics import router as lda_router
# then: app.include_router(lda_router)