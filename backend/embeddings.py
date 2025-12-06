# embeddings.py
# FastAPI router to build embeddings and run semantic search.
# Install: pip install sentence-transformers numpy
# Optional (recommended for larger scale): pip install faiss-cpu

import os
from typing import List, Tuple, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from sentence_transformers import SentenceTransformer
import numpy as np

# try to import faiss
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

from db import db, papers_collection  # assumes db.py exposes 'db' and 'papers_collection'

embeddings_collection = db["embeddings"]            # stores {doc_id, embedding: [floats]}
emb_index_meta_collection = db["emb_index_meta"]    # stores mapping & metadata
router = APIRouter()

_MODEL_NAME = os.getenv("SENTENCE_MODEL", "all-MiniLM-L6-v2")
_FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
_BATCH_SIZE = int(os.getenv("EMB_BATCH", "64"))

_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

@router.post("/embeddings/build")
def build_embeddings(batch_size: int = Query(_BATCH_SIZE, gt=0, le=1024)):
    """
    Build embeddings for documents that don't already have embeddings.
    - Encodes (title + summary) for each doc, normalizes vectors, stores them in embeddings_collection.
    - Optionally builds a FAISS index and stores mapping in emb_index_meta_collection.
    Returns summary dict.
    """
    try:
        model = get_model()

        # Fetch documents (doc_id, title, summary)
        docs = list(papers_collection.find({}, {"doc_id": 1, "title": 1, "summary": 1}))
        if not docs:
            return {"status": "error", "message": "No documents found in papers_collection"}

        # Determine which docs need embeddings
        existing_ids = set(d["doc_id"] for d in embeddings_collection.find({}, {"doc_id": 1}))
        to_process = [d for d in docs if d["doc_id"] not in existing_ids]

        # If nothing new, still update meta (doc_ids list) for FAISS
        all_doc_ids = [int(d["doc_id"]) for d in docs]

        if not to_process:
            # still ensure meta doc exists with string values
            emb_index_meta_collection.replace_one(
                {"_id": "faiss_meta"}, 
                {
                    "_id": "faiss_meta", 
                    "doc_ids": all_doc_ids,
                    "count": str(len(all_doc_ids))
                }, 
                upsert=True
            )
        else:
            texts = []
            ids_order = []
            for d in to_process:
                text = ((d.get("title") or "") + "\n" + (d.get("summary") or "")).strip()
                texts.append(text)
                ids_order.append(int(d["doc_id"]))

            # encode in batches
            embeddings_batches = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                emb = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
                embeddings_batches.append(emb.astype("float32"))
            embeddings = np.vstack(embeddings_batches)
            embeddings = l2_normalize_rows(embeddings)

            # write to Mongo per doc
            for idx, doc_id in enumerate(ids_order):
                vec = embeddings[idx].tolist()
                embeddings_collection.replace_one({"doc_id": int(doc_id)}, {"doc_id": int(doc_id), "embedding": vec}, upsert=True)

        # Now rebuild FAISS index if faiss present (optional)
        # We collect all embeddings in the same order doc_ids to create the index
        cursor = embeddings_collection.find({}, {"doc_id": 1, "embedding": 1})
        emb_list = []
        doc_ids = []
        for d in cursor:
            doc_ids.append(int(d["doc_id"]))
            emb_list.append(d["embedding"])
        if not emb_list:
            return {"status": "ok", "processed": len(to_process), "total": len(all_doc_ids), "faiss": False, "message": "No embeddings to process"}

        emb_matrix = np.array(emb_list, dtype="float32")
        # ensure normalized (should already be normalized)
        emb_matrix = l2_normalize_rows(emb_matrix)

        if _HAS_FAISS:
            dim = emb_matrix.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(emb_matrix)
            faiss.write_index(index, _FAISS_INDEX_PATH)
            emb_index_meta_collection.replace_one(
                {"_id": "faiss_meta"}, 
                {
                    "_id": "faiss_meta", 
                    "doc_ids": doc_ids, 
                    "dim": str(dim), 
                    "count": str(len(doc_ids)), 
                    "path": _FAISS_INDEX_PATH
                }, 
                upsert=True
            )
            return {"status": "ok", "processed": len(to_process), "total": len(doc_ids), "faiss": True, "dim": dim, "message": f"Built embeddings for {len(to_process)} documents with FAISS index"}
        else:
            emb_index_meta_collection.replace_one(
                {"_id": "faiss_meta"}, 
                {
                    "_id": "faiss_meta", 
                    "doc_ids": doc_ids, 
                    "dim": str(emb_matrix.shape[1]), 
                    "count": str(len(doc_ids)), 
                    "path": None
                }, 
                upsert=True
            )
            return {"status": "ok", "processed": len(to_process), "total": len(doc_ids), "faiss": False, "dim": emb_matrix.shape[1], "message": f"Built embeddings for {len(to_process)} documents (FAISS not available)"}
    
    except Exception as e:
        return {"status": "error", "message": f"Failed to build embeddings: {str(e)}"}

@router.get("/search/semantic")
def semantic_search(q: str, k: int = Query(10, gt=0, le=200)):
    """
    Semantic search:
      - compute query embedding and normalize
      - if FAISS index exists and available, use it
      - otherwise load embeddings from Mongo and compute dot products (brute-force)
    Returns top-k docs with scores and metadata.
    """
    if not q or not q.strip():
        return {"query": q, "results": []}
    model = get_model()
    q_emb = model.encode([q], convert_to_numpy=True).astype("float32")
    q_emb = l2_normalize_rows(q_emb)

    meta = emb_index_meta_collection.find_one({"_id": "faiss_meta"})
    if meta:
        doc_ids = meta.get("doc_ids", [])
        path = meta.get("path")
    else:
        doc_ids = []
        path = None

    results = []
    # FAISS fast path
    if path and _HAS_FAISS and os.path.exists(path):
        index = faiss.read_index(path)
        D, I = index.search(q_emb, k)
        D = D[0].tolist()
        I = I[0].tolist()
        for score, idx in zip(D, I):
            if idx < 0 or idx >= len(doc_ids):
                continue
            doc_id = int(doc_ids[idx])
            doc = papers_collection.find_one({"doc_id": doc_id}, {"_id": 0, "doc_id": 1, "title": 1, "summary": 1, "authors": 1, "published": 1, "link": 1})
            if not doc:
                continue
            results.append({
                "doc_id": doc_id,
                "title": doc.get("title"),
                "snippet": (doc.get("summary") or "")[:300],
                "score": float(score),
                "published": doc.get("published"),
                "authors": doc.get("authors"),
                "link": doc.get("link")
            })
        return {"query": q, "k": k, "method": "semantic", "results": results}

    # fallback: brute-force using embeddings stored in Mongo
    cursor = embeddings_collection.find({}, {"doc_id": 1, "embedding": 1})
    doc_ids_b = []
    emb_list = []
    for d in cursor:
        doc_ids_b.append(int(d["doc_id"]))
        emb_list.append(d["embedding"])
    if not emb_list:
        return {"query": q, "k": k, "method": "semantic", "results": []}
    emb_matrix = np.array(emb_list, dtype="float32")
    # ensure normalized
    emb_matrix = l2_normalize_rows(emb_matrix)
    scores = (emb_matrix @ q_emb[0])  # dot product
    topk_idx = np.argsort(-scores)[:k]
    for idx in topk_idx:
        doc_id = int(doc_ids_b[idx])
        score = float(scores[idx])
        doc = papers_collection.find_one({"doc_id": doc_id}, {"_id": 0, "doc_id": 1, "title": 1, "summary": 1, "authors": 1, "published": 1, "link": 1})
        if not doc:
            continue
        results.append({
            "doc_id": doc_id,
            "title": doc.get("title"),
            "snippet": (doc.get("summary") or "")[:300],
            "score": score,
            "published": doc.get("published"),
            "authors": doc.get("authors"),
            "link": doc.get("link")
        })
    return {"query": q, "k": k, "method": "semantic", "results": results}

# Export router to include in main.py