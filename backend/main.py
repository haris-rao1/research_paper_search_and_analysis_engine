# Updated main.py — adds relevance feedback (Rocchio) endpoint
# Replace or merge with your existing main.py. This file keeps previous /search behavior
# and adds a new POST /feedback endpoint that accepts selected doc IDs and returns
# a re-ranked result set after applying Rocchio relevance feedback.

from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Tuple, Dict, Any
import requests
import xmltodict
from datetime import datetime
import math
from collections import Counter, defaultdict

from db import papers_collection, inverted_index_collection, get_next_doc_id
from preprocessor import preprocessing
from invertedIndex import build_inverted_index

# scoring modules (assumes these files exist in your project)
from bm25 import bm25_score_for_query
from tfIdf import tfidf_score_for_query
from cosineSimilarity import cosine_similarity_for_query
from embeddings import router as embeddings_router
from lda_topics import router as lda_router
from clusters import router as clusters_router
from search_clustered import router as clustered_search_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(lda_router)
app.include_router(embeddings_router)
app.include_router(clusters_router)
app.include_router(clustered_search_router) 
# ---------------------------------------------------------------------
# Helper: save inverted index to MongoDB
# ---------------------------------------------------------------------
def save_inverted_index_to_db(inverted_index: dict, doc_lengths: dict, avg_doc_length: float):
    """
    Save the inverted index to MongoDB as a single document.
    Format: [term, df, [[doc_id, tf], ...]]
    """
    index_array = []
    for term, term_data in inverted_index.items():
        # term_data is {df: X, postings: {doc_id: tf}}
        df = term_data["df"]
        postings_dict = term_data["postings"]
        # Convert postings dict to list format with string doc_ids for MongoDB
        postings_list = [[str(doc_id), tf] for doc_id, tf in postings_dict.items()]
        index_array.append([term, df, postings_list])
    
    # Store with metadata - ensure all dict keys are strings
    inverted_index_collection.replace_one(
        {"_id": "inverted_index"},
        {
            "_id": "inverted_index",
            "index": index_array,
            "meta": {
                "doc_lengths": {str(k): int(v) for k, v in doc_lengths.items()},
                "total_docs": int(len(doc_lengths)),
                "avg_doc_length": float(avg_doc_length)
            }
        },
        upsert=True
    ) 

# ---------------------------------------------------------------------
# /papers endpoint — fetch from ArXiv, store papers, build inverted index
# ---------------------------------------------------------------------
@app.get("/papers")
def fetch_papers():
    """
    Fetch CS papers from ArXiv API, assign sequential doc_ids,
    store in MongoDB, and build inverted index from summaries.
    """
    # ArXiv API URL for CS papers
    ARXIV_API_URL = "http://export.arxiv.org/api/query?search_query=cat:cs.*&start=0&max_results=1000"
    
    try:
        response = requests.get(ARXIV_API_URL)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Failed to fetch from ArXiv: {str(e)}"}
    
    # Parse XML response
    data = xmltodict.parse(response.content)
    entries = data.get("feed", {}).get("entry", [])
    
    # Handle single entry case (xmltodict returns dict instead of list)
    if isinstance(entries, dict):
        entries = [entries]
    
    if not entries:
        return {"message": "No papers found from ArXiv API"}
    
    # Process and store each paper with sequential doc_id
    stored_papers = []
    skipped_papers = []
    docs_for_index = {}  # {doc_id: summary} for building inverted index
    
    for entry in entries:
        # Extract link first to check for duplicates
        links = entry.get("link", [])
        if isinstance(links, dict):
            links = [links]
        paper_link = ""
        for link in links:
            if link.get("@type") == "text/html" or link.get("@rel") == "alternate":
                paper_link = link.get("@href", "")
                break
        if not paper_link and links:
            paper_link = links[0].get("@href", "")
        
        # Check if paper already exists (by link or arxiv_id)
        arxiv_id = entry.get("id", "")
        existing = papers_collection.find_one({"$or": [{"link": paper_link}, {"arxiv_id": arxiv_id}]})
        if existing:
            skipped_papers.append(entry.get("title", "").replace("\n", " ").strip()[:50])
            continue
        
        # Get next sequential doc_id
        doc_id = get_next_doc_id()
        
        # Extract authors (can be single dict or list of dicts)
        authors_data = entry.get("author", [])
        if isinstance(authors_data, dict):
            authors_data = [authors_data]
        authors = [a.get("name", "") for a in authors_data]
        
        # Build paper document
        paper = {
            "doc_id": doc_id,
            "title": entry.get("title", "").replace("\n", " ").strip(),
            "summary": entry.get("summary", "").replace("\n", " ").strip(),
            "authors": authors,
            "published": entry.get("published", "")[:10],  # YYYY-MM-DD
            "link": paper_link,
            "arxiv_id": arxiv_id
        }
        
        # Store in MongoDB
        papers_collection.insert_one(paper)
        stored_papers.append({"doc_id": doc_id, "title": paper["title"]})
        
        # Add summary to docs for index building
        docs_for_index[doc_id] = paper["summary"]
    
    # Only rebuild index if we have new papers
    index_stats = None
    if stored_papers:
        # Fetch ALL papers from database to rebuild complete inverted index
        all_papers = list(papers_collection.find({}, {"doc_id": 1, "summary": 1}))
        all_docs_for_index = {paper["doc_id"]: paper["summary"] for paper in all_papers}
        
        # Preprocess ALL summaries and build complete inverted index
        processed_docs = preprocessing(all_docs_for_index)
        inverted_index, doc_lengths, avg_doc_length = build_inverted_index(processed_docs)
        
        # Save inverted index to MongoDB
        save_inverted_index_to_db(inverted_index, doc_lengths, avg_doc_length)
        
        # Clear cache so next search loads fresh index
        global _index_cache
        _index_cache = {}
        
        index_stats = {
            "unique_terms": len(inverted_index),
            "total_docs": len(doc_lengths),
            "avg_doc_length": round(avg_doc_length, 2)
        }
    
    return {
        "success": True,
        "message": f"Fetched {len(entries)} papers: {len(stored_papers)} new, {len(skipped_papers)} already exist",
        "count": len(entries),
        "inserted": len(stored_papers),
        "skipped": len(skipped_papers),
        "papers": stored_papers,
        "index_stats": index_stats
    }

# ---------------------------------------------------------------------
# Helpers: load & persist index (keeps the previous load_index_from_db)
# ---------------------------------------------------------------------
_index_cache: Dict[str, Any] = {}

def load_index_from_db(force_reload: bool = False) -> Tuple[Dict[str, Dict], Dict[int, int], int, float]:
    """
    Load the inverted index from DB and return:
      inverted_index (dict), doc_lengths (dict), total_docs, avg_doc_length
    """
    global _index_cache
    if _index_cache and not force_reload:
        return _index_cache["inv"], _index_cache["doc_lengths"], _index_cache["total_docs"], _index_cache["avgdl"]

    doc = inverted_index_collection.find_one({"_id": "inverted_index"})
    if not doc:
        return {}, {}, 0, 0.0

    inv = {}
    for item in doc.get("index", []):
        term = item[0]
        df = item[1]
        postings_list = item[2]
        # convert postings list [[doc_id, tf], ...] to dict {int(doc_id): tf}
        postings = {int(p[0]): p[1] for p in postings_list}
        inv[term] = {"df": df, "postings": postings}

    meta = doc.get("meta", {})
    doc_lengths = {int(k): int(v) for k, v in meta.get("doc_lengths", {}).items()} if meta.get("doc_lengths") else {}
    total_docs = int(meta.get("total_docs", len(doc_lengths)))
    avgdl = float(meta.get("avg_doc_length", sum(doc_lengths.values()) / total_docs if total_docs else 0.0))

    _index_cache = {"inv": inv, "doc_lengths": doc_lengths, "total_docs": total_docs, "avgdl": avgdl}
    return inv, doc_lengths, total_docs, avgdl

# ---------------------------------------------------------------------
# Helpers: build IDF map and query/doc vectors used by Rocchio
# ---------------------------------------------------------------------
def build_idf_map(inverted_index: Dict[str, Dict], total_docs: int) -> Dict[str, float]:
    idf_map = {}
    if total_docs <= 0:
        return idf_map
    for term, data in inverted_index.items():
        df = data.get("df", 0)
        # IDF formula chosen consistently with TF-IDF implementation
        idf_map[term] = math.log(total_docs / (df + 1)) + 1.0
    return idf_map

def build_query_vector_from_text(q: str, inverted_index: Dict[str, Dict], total_docs: int) -> Dict[str, float]:
    """
    Preprocess the text (using your preprocessing pipeline) and build a TF-IDF
    weighted query vector: {term: weight}
    """
    processed = preprocessing({0: q})
    query_terms = processed.get(0, [])
    if not query_terms:
        return {}

    qtf = Counter(query_terms)
    qlen = len(query_terms)
    idf_map = build_idf_map(inverted_index, total_docs)
    q_vector = {}
    for term, tf in qtf.items():
        tf_norm = tf / qlen
        idf = idf_map.get(term, math.log(total_docs / 1.0) + 1.0 if total_docs > 0 else 1.0)
        q_vector[term] = tf_norm * idf
    return q_vector

def compute_selected_docs_centroid(selected_doc_ids: List[int], inverted_index: Dict[str, Dict],
                                   doc_lengths: Dict[int, int], total_docs: int) -> Dict[str, float]:
    """
    Sum TF-IDF document vectors for selected docs and return centroid (sum vector).
    Each doc vector weight: (tf / doc_len) * idf(term)
    Returns: dict term -> summed weight (not yet averaged by |Dr|)
    """
    idf_map = build_idf_map(inverted_index, total_docs)
    centroid = defaultdict(float)
    # For each term in inverted_index, check if any selected docs contain it.
    # For small selected_doc_ids (<=5) this is acceptable.
    for term, data in inverted_index.items():
        postings = data.get("postings", {})
        idf = idf_map.get(term, 1.0)
        for doc_id in selected_doc_ids:
            tf = postings.get(doc_id, 0)
            if tf:
                dl = doc_lengths.get(doc_id, 0)
                if dl == 0:
                    continue
                doc_weight = (tf / dl) * idf
                centroid[term] += doc_weight
    return dict(centroid)

def rocchio_expand(q_vector: Dict[str, float], selected_centroid: Dict[str, float],
                   alpha: float = 1.0, beta: float = 0.75, selected_count: int = 1) -> Dict[str, float]:
    """
    Apply Rocchio formula (ignoring non-relevant set):
      q_new = alpha * q + (beta / |Dr|) * sum_{d in Dr} v_d
    q_vector and selected_centroid are dicts term->weight
    """
    q_new = defaultdict(float)
    # alpha * q
    for t, w in q_vector.items():
        q_new[t] += alpha * w
    # add beta * centroid / |Dr|
    if selected_count > 0:
        scale = beta / selected_count
        for t, w in selected_centroid.items():
            q_new[t] += scale * w
    return dict(q_new)

def top_terms_from_vector(q_vector: Dict[str, float], top_n: int = 10) -> List[str]:
    # return the top_n terms sorted by weight
    items = sorted(q_vector.items(), key=lambda x: x[1], reverse=True)
    return [t for t, _ in items[:top_n]]

def score_with_query_vector(q_vector: Dict[str, float],
                            inverted_index: Dict[str, Dict],
                            doc_lengths: Dict[int, int],
                            total_docs: int,
                            top_k: int = 10) -> List[Tuple[int, float]]:
    """
    Score documents against an arbitrary query vector (term -> weight) using cosine similarity.
    We'll compute dot products by iterating postings lists for terms in q_vector.
    Document TF-IDF weight for term t: (tf / dl) * idf(t)
    This uses idf computed from inverted_index/total_docs.
    """
    if not q_vector or total_docs <= 0:
        return []

    # Precompute idf map for the terms present in the q_vector to save work
    idf_map = build_idf_map(inverted_index, total_docs)
    q_mag = math.sqrt(sum(w*w for w in q_vector.values()))
    if q_mag == 0:
        return []

    dot = defaultdict(float)
    doc_sq = defaultdict(float)  # partial squared magnitudes (over terms in q_vector)

    for term, q_w in q_vector.items():
        entry = inverted_index.get(term)
        if not entry:
            continue
        postings = entry.get("postings", {})
        idf = idf_map.get(term, 1.0)
        for doc_id, tf in postings.items():
            dl = doc_lengths.get(doc_id, 0)
            if dl == 0:
                continue
            doc_w = (tf / dl) * idf
            dot[doc_id] += q_w * doc_w
            doc_sq[doc_id] += doc_w * doc_w

    # build scores
    scores = {}
    for doc_id, dp in dot.items():
        d_mag = math.sqrt(doc_sq.get(doc_id, 0.0))
        if d_mag == 0:
            continue
        scores[doc_id] = dp / (q_mag * d_mag)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# ---------------------------------------------------------------------
# Existing /search endpoint (unchanged) - assume present in your app
# ---------------------------------------------------------------------
@app.get("/search")
def search(
    q: str,
    method: str = Query("bm25", pattern="^(bm25|tfidf|cosine|hybrid)$"),
    k: int = 10,
    start_date: Optional[str] = None,  # "YYYY-MM-DD"
    end_date: Optional[str] = None,    # "YYYY-MM-DD"
    author: Optional[str] = None,
    hybrid_weights: Optional[str] = None
):
    processed = preprocessing({0: q})
    query_terms = processed.get(0, [])
    if not query_terms:
        return {"query": q, "results": []}

    inverted_index, doc_lengths, total_docs, avgdl = load_index_from_db()
    if total_docs == 0:
        return {"query": q, "results": [], "message": "No index available. Build index first."}

    results_bm25 = []
    results_tfidf = []
    results_cosine = []

    if method in ("bm25", "hybrid"):
        results_bm25 = bm25_score_for_query(query_terms, inverted_index, doc_lengths, total_docs, avgdl, top_k=k)
    if method in ("tfidf", "hybrid"):
        results_tfidf = tfidf_score_for_query(query_terms, inverted_index, doc_lengths, total_docs, top_k=k)
    if method in ("cosine", "hybrid"):
        results_cosine = cosine_similarity_for_query(query_terms, inverted_index, doc_lengths, total_docs, top_k=k)

    final_scores: Dict[int, float] = {}

    if method == "bm25":
        final_scores = {doc: score for doc, score in results_bm25}
    elif method == "tfidf":
        final_scores = {doc: score for doc, score in results_tfidf}
    elif method == "cosine":
        final_scores = {doc: score for doc, score in results_cosine}
    else:  # hybrid
        def to_dict(r):
            return {d: s for d, s in r}

        bm = to_dict(results_bm25)
        tf = to_dict(results_tfidf)
        cs = to_dict(results_cosine)

        max_b = max(bm.values()) if bm else 1.0
        max_t = max(tf.values()) if tf else 1.0
        max_c = max(cs.values()) if cs else 1.0

        w_b, w_t, w_c = (1/3, 1/3, 1/3)
        if hybrid_weights:
            try:
                parts = [float(x) for x in hybrid_weights.split(",")]
                if len(parts) == 3 and abs(sum(parts) - 1.0) < 1e-6:
                    w_b, w_t, w_c = parts
            except Exception:
                pass

        doc_ids = set(list(bm.keys()) + list(tf.keys()) + list(cs.keys()))
        for d in doc_ids:
            sb = (bm.get(d, 0.0) / max_b) if max_b else 0.0
            st = (tf.get(d, 0.0) / max_t) if max_t else 0.0
            sc = (cs.get(d, 0.0) / max_c) if max_c else 0.0
            final_scores[d] = w_b * sb + w_t * st + w_c * sc

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    doc_ids = [d for d, _ in ranked]

    query_filter = {"doc_id": {"$in": doc_ids}}
    if author:
        query_filter["authors"] = {"$regex": author, "$options": "i"}
    if start_date or end_date:
        date_filter = {}
        if start_date:
            date_filter["$gte"] = start_date
        if end_date:
            date_filter["$lte"] = end_date
        query_filter["published"] = date_filter

    final_docs = list(papers_collection.find(query_filter, {"_id": 0, "doc_id": 1, "title": 1, "summary": 1, "published": 1, "authors": 1, "link": 1}))
    docs_map = {doc["doc_id"]: doc for doc in final_docs}

    results = []
    for doc_id, score in ranked:
        if doc_id not in docs_map:
            continue
        doc = docs_map[doc_id]
        snippet = (doc.get("summary") or "")[:300]
        results.append({"doc_id": doc_id, "title": doc.get("title"), "score": float(score), "snippet": snippet, "published": doc.get("published"), "authors": doc.get("authors"), "link": doc.get("link")})

    return {"query": q, "method": method, "k": k, "results": results}

# ---------------------------------------------------------------------
# NEW: Relevance feedback (Rocchio) endpoint
# ---------------------------------------------------------------------
@app.post("/feedback")
def relevance_feedback(
    q: str = Body(..., embed=True),
    selected_doc_ids: List[int] = Body(..., embed=True),
    k: int = Body(10, embed=True),
    method: str = Body("cosine", embed=True),  # scoring method to use after expansion
    alpha: float = Body(1.0, embed=True),
    beta: float = Body(0.75, embed=True),
    top_terms: int = Body(10, embed=True)
):
    """
    Relevance feedback using Rocchio:
      - q: original query string
      - selected_doc_ids: list of doc_id ints that user marked relevant (1..5 typically)
      - k: number of results to return
      - method: which scoring method to use after expansion (cosine/tfidf/bm25/hybrid)
      - alpha/beta: Rocchio parameters (gamma for non-relevant not implemented)
      - top_terms: number of top weighted terms from expanded query to use as new query
    Returns: expanded_query_terms and re-ranked results
    """
    # load index and metadata
    inverted_index, doc_lengths, total_docs, avgdl = load_index_from_db()
    if total_docs == 0:
        return {"error": "No index available. Build index first."}

    # 1) build original query vector
    q_vector = build_query_vector_from_text(q, inverted_index, total_docs)

    # 2) build centroid of selected docs
    if not selected_doc_ids:
        return {"error": "No selected documents provided."}
    selected_centroid = compute_selected_docs_centroid(selected_doc_ids, inverted_index, doc_lengths, total_docs)

    # 3) apply Rocchio
    q_new_vector = rocchio_expand(q_vector, selected_centroid, alpha=alpha, beta=beta, selected_count=len(selected_doc_ids))

    # 4) derive top terms from q_new_vector to form an expanded query (simple approach)
    expanded_terms = top_terms_from_vector(q_new_vector, top_n=top_terms)
    expanded_query_text = " ".join(expanded_terms)  # for display/debug

    # 5) Score using the expanded q_vector (direct scoring) or use expanded_terms as query to existing search
    # We'll use direct scoring on q_new_vector (cosine-like) for best fidelity, but we also support fallback to method param.
    ranked = score_with_query_vector(q_new_vector, inverted_index, doc_lengths, total_docs, top_k=k)

    # If method requested is not cosine, we can call other scorers by using expanded_terms as query_terms
    if method in ("bm25", "tfidf", "hybrid"):
        # call existing scorers using expanded_terms
        # For bm25/tfidf/hybrid use the existing functions which accept query_terms list
        if method == "bm25":
            ranked = bm25_score_for_query(expanded_terms, inverted_index, doc_lengths, avgdl=avgdl, total_docs=total_docs, top_k=k) if False else bm25_score_for_query(expanded_terms, inverted_index, doc_lengths, avg_doc_length=avgdl, total_docs=total_docs, top_k=k)  # keep signature compatibility; final call below
            # Note: your bm25_score_for_query signature expects (query_terms, inverted_index, doc_lengths, total_docs, avg_doc_length, k1, b, top_k)
            ranked = bm25_score_for_query(expanded_terms, inverted_index, doc_lengths, total_docs, avgdl, top_k=k)
        elif method == "tfidf":
            ranked = tfidf_score_for_query(expanded_terms, inverted_index, doc_lengths, total_docs, top_k=k)
        elif method == "hybrid":
            # compute each and combine equally
            r_bm = bm25_score_for_query(expanded_terms, inverted_index, doc_lengths, total_docs, avgdl, top_k=k)
            r_tf = tfidf_score_for_query(expanded_terms, inverted_index, doc_lengths, total_docs, top_k=k)
            r_cs = cosine_similarity_for_query(expanded_terms, inverted_index, doc_lengths, total_docs, top_k=k)
            # normalize and combine
            bm = {d: s for d, s in r_bm}
            tf = {d: s for d, s in r_tf}
            cs = {d: s for d, s in r_cs}
            max_b = max(bm.values()) if bm else 1.0
            max_t = max(tf.values()) if tf else 1.0
            max_c = max(cs.values()) if cs else 1.0
            doc_ids = set(list(bm.keys()) + list(tf.keys()) + list(cs.keys()))
            combined = {}
            for d in doc_ids:
                sb = (bm.get(d, 0.0) / max_b) if max_b else 0.0
                st = (tf.get(d, 0.0) / max_t) if max_t else 0.0
                sc = (cs.get(d, 0.0) / max_c) if max_c else 0.0
                combined[d] = (sb + st + sc) / 3.0
            ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]

    # 6) fetch metadata for ranked docs
    doc_ids = [d for d, _ in ranked]
    query_filter = {"doc_id": {"$in": doc_ids}}
    final_docs = list(papers_collection.find(query_filter, {"_id": 0, "doc_id": 1, "title": 1, "summary": 1, "published": 1, "authors": 1, "link": 1}))
    docs_map = {doc["doc_id"]: doc for doc in final_docs}

    results = []
    for doc_id, score in ranked:
        if doc_id not in docs_map:
            continue
        doc = docs_map[doc_id]
        snippet = (doc.get("summary") or "")[:300]
        results.append({"doc_id": doc_id, "title": doc.get("title"), "score": float(score), "snippet": snippet, "published": doc.get("published"), "authors": doc.get("authors"), "link": doc.get("link")})

    return {
        "original_query": q,
        "expanded_query_terms": expanded_terms,
        "expanded_query_text": expanded_query_text,
        "method_used_for_rerank": method,
        "k": k,
        "results": results
    }