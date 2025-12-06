# Research Papers IR System - Backend

A comprehensive Information Retrieval system for academic papers from ArXiv, featuring advanced search capabilities, topic modeling, clustering, and semantic search.

## 🚀 Features

- **Multi-method Search**: BM25, TF-IDF, Cosine Similarity, Semantic Search, and Hybrid Search
- **Topic Modeling**: LDA-based topic discovery with UMAP visualization
- **Document Clustering**: HDBSCAN clustering with centroid-based search
- **Semantic Search**: Sentence transformer embeddings with cosine similarity
- **Relevance Feedback**: Rocchio algorithm for query expansion
- **Cluster-First Search**: Advanced retrieval with cluster → topic → ranking pipeline

## 📋 Prerequisites

- Python 3.9 or higher
- MongoDB Atlas account (or local MongoDB instance)
- 4GB+ RAM recommended

## 🔧 Installation

### 1. Clone the Repository

```bash
cd C:\Users\HARIS RAO\Desktop\irProjects\backend
```

### 2. Create Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
# Core dependencies
pip install fastapi uvicorn[standard] pymongo python-dotenv

# Data processing
pip install requests xmltodict nltk

# Machine Learning & NLP
pip install sentence-transformers numpy scikit-learn

# Topic Modeling
pip install gensim

# Clustering
pip install hdbscan umap-learn

# Optional: For better performance with embeddings
pip install faiss-cpu
```

**Or install all at once:**

```powershell
pip install fastapi uvicorn[standard] pymongo python-dotenv requests xmltodict nltk sentence-transformers numpy scikit-learn gensim hdbscan umap-learn faiss-cpu
```

### 4. Download NLTK Data

```powershell
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### 5. Configure Environment Variables

Create a `.env` file in the backend directory:

```env
# MongoDB Connection
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

# Optional: Model Configuration
SENTENCE_MODEL=all-MiniLM-L6-v2
EMB_BATCH=64
```

**MongoDB Setup:**
1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a free cluster
3. Create a database user
4. Get your connection string and replace in `.env`

## 🏃 Running the Server

### Start the Backend Server

```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Run the server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at: `http://127.0.0.1:8000`

### API Documentation

Once running, visit:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 📚 API Endpoints

### Papers
- `GET /papers` - Fetch papers from ArXiv and build inverted index

### Search
- `GET /search?q=query&method=bm25&k=10` - Standard search
- `GET /search/semantic?q=query&k=10` - Semantic search with embeddings
- `GET /search/clustered?q=query&method=semantic&k=10` - Cluster-first search
- `POST /feedback` - Rocchio relevance feedback

### Embeddings
- `POST /embeddings/build` - Build sentence embeddings for all documents

### Topic Modeling
- `POST /topics/lda/build?num_topics=10` - Build LDA topic model
- `GET /topics/lda` - List all topics
- `GET /topics/lda/{topic_id}/docs` - Get documents for a topic
- `GET /topics/lda/umap?limit=400` - Get UMAP coordinates for visualization

### Clustering
- `POST /clusters/build?method=hdbscan&min_cluster_size=8` - Build document clusters
- `GET /clusters` - List all clusters
- `GET /clusters/{cluster_id}/docs` - Get documents in a cluster
- `GET /clusters/umap?limit=400` - Get cluster UMAP visualization

## 🗂️ Project Structure

```
backend/
├── main.py                 # FastAPI application & main endpoints
├── db.py                   # MongoDB connection & collections
├── preprocessor.py         # Text preprocessing (tokenization, stopwords)
├── invertedIndex.py        # Inverted index construction
├── bm25.py                 # BM25 scoring algorithm
├── tfIdf.py                # TF-IDF scoring
├── cosineSimilarity.py     # Cosine similarity scoring
├── embeddings.py           # Sentence embeddings & semantic search
├── lda_topics.py           # LDA topic modeling
├── clusters.py             # HDBSCAN clustering
├── search_clustered.py     # Advanced cluster-first search
└── README.md               # This file
```

## 🔄 Typical Workflow

1. **Fetch Papers**
   ```bash
   curl http://127.0.0.1:8000/papers
   ```

2. **Build Embeddings** (for semantic search)
   ```bash
   curl -X POST http://127.0.0.1:8000/embeddings/build
   ```

3. **Build Topics** (for topic modeling)
   ```bash
   curl -X POST "http://127.0.0.1:8000/topics/lda/build?num_topics=10"
   ```

4. **Build Clusters** (for clustering)
   ```bash
   curl -X POST "http://127.0.0.1:8000/clusters/build?method=hdbscan&min_cluster_size=8"
   ```

5. **Search**
   ```bash
   curl "http://127.0.0.1:8000/search?q=deep+learning&method=bm25&k=10"
   ```

## 🐛 Troubleshooting

### ModuleNotFoundError
- Ensure virtual environment is activated: `.\venv\Scripts\Activate.ps1`
- Reinstall dependencies

### MongoDB Connection Error
- Check your `MONGO_URI` in `.env`
- Ensure your IP is whitelisted in MongoDB Atlas
- Verify database user credentials

### NLTK Data Missing
```powershell
python -c "import nltk; nltk.download('all')"
```

### Slow Performance
- Ensure you're using bulk queries (recently optimized)
- Check MongoDB indexes are created (done automatically in `db.py`)
- Consider using local MongoDB for faster access

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | Latest | Web framework |
| uvicorn | Latest | ASGI server |
| pymongo | Latest | MongoDB driver |
| sentence-transformers | Latest | Semantic embeddings |
| scikit-learn | Latest | ML utilities |
| hdbscan | Latest | Clustering algorithm |
| gensim | Latest | Topic modeling |
| nltk | Latest | NLP preprocessing |
| numpy | Latest | Numerical operations |
| umap-learn | Latest | Dimensionality reduction |

## 📝 License

This project is for educational purposes.

## 👨‍💻 Developer

**HARIS RAO**