# ============================================
# db.py - MongoDB Database Connection & Setup
# ============================================
# This file handles all the database stuff.
# We're using MongoDB Atlas (cloud) to store our papers and inverted index.

import os
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError
from dotenv import load_dotenv

# MongoDB connection string - using Atlas cloud database
# You can change this to your own MongoDB URI if needed
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["research_papers_db"]

# ---- Our Collections (like tables in SQL) ----
papers_collection = db["papers"]                    # stores all the research papers
inverted_index_collection = db["inverted_index"]    # stores our inverted index
counters_collection = db["counters"]                # keeps track of doc_id sequence (1,2,3...)

# Additional shared collections (added so other modules can import them from db.py)
embeddings_collection = db["embeddings"]            # stores per-doc dense embeddings (doc_id, embedding)
emb_index_meta_collection = db["emb_index_meta"]    # metadata for FAISS or embeddings index
topics_lda_collection = db["topics_lda"]            # stores LDA topic metadata (_id: "lda_meta")
clusters_meta_collection = db["clusters_meta"]      # stores clustering metadata (_id: "clusters_meta")

# ---- Creating Indexes for faster queries ----
# These help MongoDB find documents quickly

# Make sure we don't store the same paper twice (unique link)
try:
    papers_collection.create_index([("link", ASCENDING)], unique=True)
except Exception:
    pass  # index already exists, no problem

# doc_id should also be unique
try:
    papers_collection.create_index([("doc_id", ASCENDING)], unique=True)
except Exception:
    pass

# index on term for faster lookups when searching
try:
    inverted_index_collection.create_index([("term", ASCENDING)], unique=True)
except Exception:
    pass

# index for embeddings/doc_id for faster lookups (useful for clustering/search)
try:
    embeddings_collection.create_index([("doc_id", ASCENDING)], unique=True)
except Exception:
    pass

# index cluster_id on papers to quickly fetch cluster members
try:
    papers_collection.create_index([("cluster_id", ASCENDING)])
except Exception:
    pass


def get_next_doc_id():
    """
    Returns the next document ID in sequence: 1, 2, 3, 4...
    
    We use a separate 'counters' collection to keep track of the last used ID.
    Every time we call this function, it increments the counter and returns the new value.
    This way each paper gets a unique sequential ID.
    """
    # find the counter and increment it by 1
    # upsert=True means create it if it doesn't exist
    result = counters_collection.find_one_and_update(
        {"_id": "doc_id_counter"},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True
    )
    
    # edge case: if somehow result is None, start from 1
    if result is None:
        counters_collection.insert_one({"_id": "doc_id_counter", "seq": 1})
        return 1
    
    return result["seq"]