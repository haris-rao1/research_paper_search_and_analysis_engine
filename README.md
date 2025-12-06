# Research Papers Information Retrieval System

A full-stack Information Retrieval system for academic papers from ArXiv, featuring advanced search algorithms, topic modeling, document clustering, and semantic search capabilities.

## 📖 Overview

This project is a comprehensive IR system that fetches academic papers from ArXiv, processes them, and provides multiple search and exploration methods including traditional IR algorithms (BM25, TF-IDF), semantic search with transformers, LDA topic modeling, and HDBSCAN clustering.

### Key Features

- 🔍 **Multi-Method Search**: BM25, TF-IDF, Cosine Similarity, Semantic Search, Hybrid Search
- 🧠 **Semantic Search**: Sentence transformer embeddings with cosine similarity
- 📊 **Topic Modeling**: LDA-based topic discovery with UMAP visualization
- 🎯 **Document Clustering**: HDBSCAN clustering with cluster-first search
- 🔄 **Relevance Feedback**: Rocchio algorithm for query expansion
- 📈 **Interactive Visualizations**: Plotly-based topic and cluster exploration
- 🚀 **Modern UI**: React 19 + Next.js 16 with TailwindCSS

## 🏗️ Architecture

```
irProjects/
├── backend/          # FastAPI Python backend
│   ├── main.py
│   ├── db.py
│   ├── embeddings.py
│   ├── lda_topics.py
│   ├── clusters.py
│   └── ...
├── frontend/         # Next.js React frontend
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   └── utils/
│   └── ...
└── README.md        # This file
```

### Tech Stack

**Backend:**
- FastAPI (Python web framework)
- MongoDB Atlas (cloud database)
- Sentence Transformers (embeddings)
- Gensim (LDA topic modeling)
- HDBSCAN (clustering)
- scikit-learn, NumPy (ML/data processing)

**Frontend:**
- Next.js 16 (React framework)
- React 19 (UI library)
- TailwindCSS 4 (styling)
- Plotly.js (visualizations)

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** (for backend)
- **Node.js 18+** (for frontend)
- **MongoDB Atlas account** (free tier works)

### 1. Clone the Repository

```powershell
cd C:\Users\HARIS RAO\Desktop\irProjects
```

### 2. Setup Backend

```powershell
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Or install manually:
pip install fastapi uvicorn[standard] pymongo python-dotenv requests xmltodict nltk sentence-transformers numpy scikit-learn gensim hdbscan umap-learn faiss-cpu

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Create .env file with MongoDB URI
echo "MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority" > .env

# Run backend server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend will run at: **http://127.0.0.1:8000**

### 3. Setup Frontend

```powershell
# Open new terminal and navigate to frontend
cd C:\Users\HARIS RAO\Desktop\irProjects\frontend

# Install dependencies
npm install

# (Optional) Create .env.local for API URL
echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000" > .env.local

# Run development server
npm run dev
```

Frontend will run at: **http://localhost:3000**

## 📚 Complete Workflow

### Step 1: Fetch Papers
1. Open http://localhost:3000
2. Click **"Fetch Papers"** button
3. Wait for ~1000 papers to be fetched from ArXiv (~30-60 seconds)
4. Papers are stored in MongoDB with inverted index built

### Step 2: Build Embeddings (Required for Semantic Search)
1. Click **"Build Embeddings"** button
2. Wait for sentence embeddings to be computed (~1-2 minutes)
3. Enables semantic search functionality

### Step 3: Build Topics (Optional - for Topic Modeling)
1. Click **"Build Topics"** button
2. LDA model trains on all documents (~1-3 minutes)
3. Topics appear in the Topics Panel with UMAP visualization

### Step 4: Build Clusters (Optional - for Clustering)
1. Click **"Build Clusters"** button
2. HDBSCAN clustering runs on embeddings (~1-2 minutes)
3. Clusters appear in the Topics Panel

### Step 5: Search and Explore

**Standard Search:**
1. Enter query (e.g., "deep learning neural networks")
2. Select method: BM25, TF-IDF, Cosine, Semantic, or Hybrid
3. Click **"Search"**

**Cluster-First Search:**
1. Enable **"Cluster-first search"** checkbox
2. Enter query
3. System finds relevant cluster → topic → ranks documents within cluster

**Relevance Feedback:**
1. Perform a search
2. Select relevant documents using checkboxes
3. Click **"Refine with selected"**
4. Query expanded using Rocchio algorithm

**Topic/Cluster Exploration:**
1. Browse topics/clusters in the side panel
2. Click on a topic/cluster card
3. View all documents in that topic/cluster
4. Use UMAP visualization to explore document space

## 🔌 API Endpoints

### Backend API Documentation
Once backend is running, visit:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Main Endpoints

| Category | Endpoint | Method | Description |
|----------|----------|--------|-------------|
| **Papers** | `/papers` | GET | Fetch papers from ArXiv |
| **Search** | `/search` | GET | Standard search (BM25/TF-IDF/Cosine/Hybrid) |
| | `/search/semantic` | GET | Semantic search with embeddings |
| | `/search/clustered` | GET | Cluster-first search |
| | `/feedback` | POST | Relevance feedback (Rocchio) |
| **Embeddings** | `/embeddings/build` | POST | Build sentence embeddings |
| **Topics** | `/topics/lda/build` | POST | Build LDA topic model |
| | `/topics/lda` | GET | List all topics |
| | `/topics/lda/{id}/docs` | GET | Get documents for topic |
| | `/topics/lda/umap` | GET | Get UMAP visualization |
| **Clusters** | `/clusters/build` | POST | Build document clusters |
| | `/clusters` | GET | List all clusters |
| | `/clusters/{id}/docs` | GET | Get documents in cluster |
| | `/clusters/umap` | GET | Get cluster visualization |

## 📊 Project Components

### Backend Modules

| Module | Purpose |
|--------|---------|
| `main.py` | FastAPI app, search endpoints, Rocchio feedback |
| `db.py` | MongoDB connection and collections setup |
| `preprocessor.py` | Text preprocessing (tokenization, stopwords) |
| `invertedIndex.py` | Inverted index construction |
| `bm25.py` | BM25 ranking algorithm |
| `tfIdf.py` | TF-IDF scoring |
| `cosineSimilarity.py` | Cosine similarity scoring |
| `embeddings.py` | Sentence embeddings & semantic search |
| `lda_topics.py` | LDA topic modeling & UMAP |
| `clusters.py` | HDBSCAN clustering |
| `search_clustered.py` | Cluster-first search pipeline |

### Frontend Components

| Component | Purpose |
|-----------|---------|
| `Header.jsx` | Action buttons (fetch, build embeddings/topics/clusters) |
| `SearchForm.jsx` | Search input, filters, method selection |
| `ResultList.jsx` | Display search results |
| `ResultItem.jsx` | Individual result card |
| `FeedbackBar.jsx` | Relevance feedback controls |
| `TopicsPanel.jsx` | Topics/clusters visualization with Plotly |
| `api.js` | API client functions |

## ⚡ Performance Optimizations

### Backend
- **Bulk MongoDB Queries**: Replaced N+1 queries with bulk operations (10-50x faster)
- **Efficient Indexing**: MongoDB indexes on doc_id, cluster_id, topic_id
- **Batch Processing**: Embeddings computed in batches
- **Optimized Search**: Restricted candidate sets in cluster-first search

### Frontend
- **Parallel API Calls**: Topics and clusters fetched simultaneously (4x faster)
- **Dynamic Imports**: Plotly loaded only on client-side (reduces bundle size)
- **React 19 Compiler**: Automatic optimizations
- **Lazy Loading**: Components loaded on-demand

## 🐛 Troubleshooting

### Backend Issues

**MongoDB Connection Error:**
```powershell
# Check .env file has correct MONGO_URI
# Ensure IP is whitelisted in MongoDB Atlas
# Verify database user credentials
```

**Module Not Found:**
```powershell
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

**NLTK Data Missing:**
```powershell
python -c "import nltk; nltk.download('all')"
```

### Frontend Issues

**Backend Connection Failed:**
```powershell
# Ensure backend is running at http://127.0.0.1:8000
# Check browser console for CORS errors
# Verify NEXT_PUBLIC_API_URL in .env.local
```

**Port Already in Use:**
```powershell
# Kill process on port 3000
Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess | Stop-Process

# Or use different port
$env:PORT=3001; npm run dev
```

**Module Not Found:**
```powershell
rm -r node_modules
npm install
```

## 📈 System Requirements

### Minimum
- **RAM**: 4GB
- **Storage**: 2GB free space
- **CPU**: Dual-core processor
- **Internet**: For MongoDB Atlas and ArXiv API

### Recommended
- **RAM**: 8GB+ (for faster embeddings/clustering)
- **Storage**: 5GB free space
- **CPU**: Quad-core processor
- **Internet**: Stable connection for API calls

## 🔒 Security Notes

- Never commit `.env` files with real credentials
- Use MongoDB Atlas network access controls
- Keep dependencies updated
- Use environment variables for sensitive data

## 📝 License

This project is for educational purposes.

## 👨‍💻 Developer

**HARIS RAO**

## 🤝 Contributing

This is an academic project. For questions or issues, please refer to the documentation in:
- `backend/README.md` - Backend-specific documentation
- `frontend/README.md` - Frontend-specific documentation

## 📞 Support

- **Backend API Docs**: http://127.0.0.1:8000/docs
- **Check Server Status**: Ensure both servers are running
- **Review Logs**: Check terminal output for errors

---

**Happy Searching! 🔍📚**