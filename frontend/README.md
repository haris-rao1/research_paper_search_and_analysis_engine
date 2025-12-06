# Research Papers IR System - Frontend

Modern, responsive web interface for the Research Papers Information Retrieval system built with Next.js 16, React 19, and TailwindCSS 4.

## 🎨 Features

- **Multi-Method Search**: BM25, TF-IDF, Cosine Similarity, Semantic Search, and Hybrid Search
- **Cluster-First Search**: Advanced search starting from document clusters
- **Topic Visualization**: Interactive LDA topic exploration with Plotly
- **Cluster Visualization**: UMAP-based cluster visualization
- **Relevance Feedback**: Select relevant documents to refine search results
- **Real-time Updates**: Live search results and status updates
- **Responsive Design**: Modern UI with TailwindCSS

## 📋 Prerequisites

- Node.js 18.x or higher
- npm or yarn package manager
- Backend API running on `http://127.0.0.1:8000`

## 🔧 Installation

### 1. Navigate to Frontend Directory

```powershell
cd C:\Users\HARIS RAO\Desktop\irProjects\frontend
```

### 2. Install Dependencies

```powershell
npm install
```

Or with yarn:

```powershell
yarn install
```

### 3. Configure Environment Variables (Optional)

Create a `.env.local` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

If not set, it defaults to `http://127.0.0.1:8000`

## 🏃 Running the Application

### Development Mode

```powershell
npm run dev
```

The application will be available at: `http://localhost:3000`

### Production Build

```powershell
# Build for production
npm run build

# Start production server
npm start
```

### Linting

```powershell
npm run lint
```

## 🗂️ Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── globals.css           # Global styles
│   │   ├── layout.js             # Root layout component
│   │   ├── page.js               # Home page (redirects to display)
│   │   └── display/
│   │       └── page.jsx          # Main application page
│   ├── components/
│   │   ├── Header.jsx            # Header with action buttons
│   │   ├── SearchForm.jsx        # Search form with filters
│   │   ├── ResultList.jsx        # Search results display
│   │   ├── ResultItem.jsx        # Individual result card
│   │   ├── FeedbackBar.jsx       # Relevance feedback controls
│   │   └── TopicsPanel.jsx       # Topics & clusters visualization
│   └── utils/
│       └── api.js                # API client functions
├── public/                       # Static assets
├── package.json                  # Dependencies and scripts
├── next.config.mjs              # Next.js configuration
├── tailwind.config.js           # TailwindCSS configuration
├── postcss.config.mjs           # PostCSS configuration
└── README.md                    # This file
```

## 🎯 How to Use

### 1. Fetch Papers from ArXiv

1. Click **"Fetch Papers"** button in the header
2. Wait for papers to be fetched and indexed (~30-60 seconds for 1000 papers)
3. Status message will confirm success

### 2. Build Embeddings (for Semantic Search)

1. Click **"Build Embeddings"** button
2. Wait for embeddings to be computed (~1-2 minutes)
3. Status will show completion

### 3. Build LDA Topics

1. Click **"Build Topics"** button
2. LDA model will be trained on all documents
3. Topics will appear in the Topics Panel

### 4. Build Clusters

1. Click **"Build Clusters"** button
2. HDBSCAN clustering will run on embeddings
3. Clusters will appear in the Topics Panel

### 5. Search

**Basic Search:**
1. Enter query in search box
2. Select search method (BM25, TF-IDF, Cosine, Semantic, Hybrid)
3. Set number of results (k)
4. Click **"Search"**

**Cluster-First Search:**
1. Check **"Cluster-first search"** checkbox
2. Enter query and click Search
3. System will find relevant cluster → topic → rank documents

**Filters:**
- Author: Filter by author name
- Start/End Date: Filter by publication date
- Hybrid Weights: For hybrid search (e.g., "0.5,0.25,0.25")

### 6. Relevance Feedback

1. Perform a search
2. Click checkboxes to select relevant documents
3. Click **"Refine with selected"** button
4. System will expand query using Rocchio algorithm

### 7. Explore Topics & Clusters

1. Switch between "Topics" and "Clusters" tabs in the panel
2. Click on a topic/cluster card to view its documents
3. Use the UMAP visualization to explore document space
4. Click **"Show"** button to load selected topic/cluster documents

## 🎨 UI Components

### Header
- Fetch Papers button
- Build Embeddings button
- Build Topics button
- Build Clusters button
- Status indicators

### Search Form
- Query input
- Method selector (BM25, TF-IDF, Cosine, Semantic, Hybrid)
- Results count (k)
- Cluster-first mode toggle
- Author filter
- Date range filters
- Hybrid weights input (for hybrid search)

### Results List
- Document cards with title, snippet, score
- Authors and publication date
- Link to ArXiv paper
- Checkbox for relevance feedback

### Topics Panel
- Topics/Clusters toggle
- Grid of topic/cluster cards
- Interactive UMAP visualization (Plotly)
- Clear and Show buttons

### Feedback Bar
- Selected documents count
- Refine button
- Clear selection button

## 🔌 API Integration

The frontend communicates with the backend through these endpoints:

| Function | Endpoint | Method |
|----------|----------|--------|
| Fetch papers | `/papers` | GET |
| Standard search | `/search` | GET |
| Semantic search | `/search/semantic` | GET |
| Cluster search | `/search/clustered` | GET |
| Relevance feedback | `/feedback` | POST |
| Build embeddings | `/embeddings/build` | POST |
| Build topics | `/topics/lda/build` | POST |
| Build clusters | `/clusters/build` | POST |
| Get topics | `/topics/lda` | GET |
| Get clusters | `/clusters` | GET |

All API calls are defined in `src/utils/api.js`

## 🐛 Troubleshooting

### Backend Connection Failed
- Ensure backend server is running on `http://127.0.0.1:8000`
- Check `NEXT_PUBLIC_API_URL` environment variable
- Verify CORS is enabled on backend

### Plotly Visualization Not Showing
- This is normal on first load (dynamic import)
- Refresh the page if visualization doesn't appear

### Search Returns No Results
- Ensure papers are fetched first
- Check if embeddings are built (for semantic search)
- Verify clusters/topics are built (for cluster-first search)

### Module Not Found Errors
```powershell
# Delete node_modules and reinstall
rm -r node_modules
npm install
```

### Port Already in Use
```powershell
# Kill process on port 3000
Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess | Stop-Process

# Or use a different port
$env:PORT=3001; npm run dev
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| next | 16.0.6 | React framework with SSR |
| react | 19.2.0 | UI library |
| react-dom | 19.2.0 | React DOM renderer |
| react-plotly.js | ^2.6.0 | Interactive visualizations |
| plotly.js-basic-dist | ^3.3.0 | Plotly library (lightweight) |
| tailwindcss | ^4 | Utility-first CSS framework |

## 🚀 Performance Optimizations

- **Dynamic Imports**: Plotly loaded only on client-side
- **Parallel API Calls**: Topics and clusters fetched simultaneously
- **Optimized Rendering**: React 19 with compiler optimizations
- **Lazy Loading**: Components loaded on-demand

## 📝 License

This project is for educational purposes.

## 👨‍💻 Developer

**HARIS RAO**

---

**Need Help?** 
- Check backend is running: `http://127.0.0.1:8000/docs`
- Review browser console for errors (F12)
- Ensure all dependencies are installed