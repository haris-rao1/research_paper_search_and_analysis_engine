'use client'
import { useState } from 'react';

import Header from '@/components/Header';
import SearchForm from '@/components/SearchForm';
import ResultsList from '@/components/ResultList';
import FeedbackBar from '@/components/FeedbackBar';
import TopicsPanel from '@/components/TopicsPanel';

import {
  fetchPapersApi,
  searchApi,
  feedbackApi,
  buildEmbeddingsApi,
  buildLdaTopicsApi,
  buildClustersApi,
  searchClusteredApi,
  getTopicDocs,
  getClusterDocs
} from '@/utils/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export default function Display() {
  // Fetch papers state
  const [fetchStatus, setFetchStatus] = useState(null);
  const [fetchLoading, setFetchLoading] = useState(false);
  const [fetchError, setFetchError] = useState(null);

  // Build embeddings state
  const [buildLoading, setBuildLoading] = useState(false);
  const [buildStatus, setBuildStatus] = useState(null);
  const [buildError, setBuildError] = useState(null);

  // Build LDA topics state
  const [buildLdaLoading, setBuildLdaLoading] = useState(false);
  const [buildLdaStatus, setBuildLdaStatus] = useState(null);
  const [buildLdaError, setBuildLdaError] = useState(null);

  // Build Clusters state
  const [buildClustersLoading, setBuildClustersLoading] = useState(false);
  const [buildClustersStatus, setBuildClustersStatus] = useState(null);
  const [buildClustersError, setBuildClustersError] = useState(null);

  // Search state
  const [query, setQuery] = useState('');
  const [method, setMethod] = useState('bm25');
  const [k, setK] = useState(10);
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [author, setAuthor] = useState('');
  const [hybridWeights, setHybridWeights] = useState('0.33,0.33,0.34');

  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [results, setResults] = useState([]);

  // Cluster-first / predicted meta
  const [clusterMode, setClusterMode] = useState(true);
  const [predictedCluster, setPredictedCluster] = useState(null);
  const [predictedClusterScore, setPredictedClusterScore] = useState(null);
  const [predictedTopic, setPredictedTopic] = useState(null);

  // Relevance feedback state
  const [selectedDocs, setSelectedDocs] = useState(new Set());
  const [refineLoading, setRefineLoading] = useState(false);

  const parseHybridWeights = (s) => {
    if (!s) return null;
    const parts = s.split(',').map(p => parseFloat(p.trim())).filter(n => !Number.isNaN(n));
    if (parts.length !== 3) return null;
    const sum = parts.reduce((a,b) => a+b, 0);
    if (Math.abs(sum - 1.0) > 1e-6) return null;
    return parts;
  };

  // Fetch papers from backend and build index
  const fetchPapers = async () => {
    setFetchLoading(true);
    setFetchError(null);
    setFetchStatus(null);

    try {
      const data = await fetchPapersApi(API_URL);
      setFetchLoading(false);
      setFetchStatus({
        count: data.count,
        inserted: data.inserted,
        success: true
      });
    } catch (err) {
      console.error('Error fetchPapers:', err);
      setFetchError('Failed to fetch papers. See console for details.');
      setFetchLoading(false);
    }
  };

  // Build embeddings for docs (calls POST /embeddings/build)
  const buildEmbeddings = async () => {
    setBuildLoading(true);
    setBuildError(null);
    setBuildStatus(null);

    try {
      const data = await buildEmbeddingsApi(API_URL);
      setBuildLoading(false);
      setBuildStatus({ success: true, info: data });
    } catch (err) {
      console.error('Error buildEmbeddings:', err);
      setBuildError('Failed to build embeddings. See console for details.');
      setBuildLoading(false);
    }
  };

  // Build LDA topics (calls POST /topics/lda/build)
  const buildLdaTopics = async (params = {}) => {
    setBuildLdaLoading(true);
    setBuildLdaError(null);
    setBuildLdaStatus(null);

    try {
      const data = await buildLdaTopicsApi(API_URL, params);
      setBuildLdaLoading(false);
      setBuildLdaStatus({ success: true, info: data });
    } catch (err) {
      console.error('Error buildLdaTopics:', err);
      setBuildLdaError('Failed to build LDA topics. See console for details.');
      setBuildLdaLoading(false);
    }
  };

  // Build clusters (calls POST /clusters/build)
  const buildClusters = async (params = {}) => {
    setBuildClustersLoading(true);
    setBuildClustersError(null);
    setBuildClustersStatus(null);

    try {
      const data = await buildClustersApi(API_URL, params);
      setBuildClustersLoading(false);
      setBuildClustersStatus({ success: true, info: data });
    } catch (err) {
      console.error('Error buildClusters:', err);
      setBuildClustersError('Failed to build clusters. See console for details.');
      setBuildClustersLoading(false);
    }
  };

  // Perform search
  const runSearch = async (e) => {
    e && e.preventDefault();
    setSearchLoading(true);
    setSearchError(null);
    setResults([]);
    setSelectedDocs(new Set());
    setPredictedCluster(null);
    setPredictedTopic(null);
    setPredictedClusterScore(null);

    if (!query || query.trim().length === 0) {
      setSearchError('Please enter a query.');
      setSearchLoading(false);
      return;
    }

    if (method === 'hybrid') {
      const parts = parseHybridWeights(hybridWeights);
      if (!parts) {
        setSearchError('Hybrid weights must be three comma-separated numbers that sum to 1, e.g. 0.5,0.25,0.25');
        setSearchLoading(false);
        return;
      }
    }

    try {
      if (clusterMode) {
        // Use cluster-first endpoint
        const params = {
          q: query,
          method,
          k,
          cluster_mode: true,
          cluster_top_m: 1,
          cluster_threshold: 0.45
        };
        const data = await searchClusteredApi(API_URL, params);
        setSearchLoading(false);
        setResults(Array.isArray(data.results) ? data.results : []);

        // set predicted cluster / topic if returned
        if (data.selected_cluster_meta) {
          setPredictedCluster(data.selected_cluster_meta);
        }
        if (data.cluster_scores && data.cluster_scores.length) {
          setPredictedClusterScore(data.cluster_scores[0]);
        }
        if (data.predicted_topic) {
          setPredictedTopic(data.predicted_topic);
        }

        if (!data.results || data.results.length === 0) {
          setSearchError('No results found for this query.');
        }
      } else {
        // default old behavior
        const params = {
          q: query,
          method,
          k,
          start_date: startDate,
          end_date: endDate,
          author,
          hybrid_weights: method === 'hybrid' ? hybridWeights : undefined
        };
        const data = await searchApi(API_URL, params);
        setSearchLoading(false);
        setResults(Array.isArray(data.results) ? data.results : []);
        if (!data.results || data.results.length === 0) {
          setSearchError('No results found for this query.');
        }
      }
    } catch (err) {
      console.error('Search error:', err);
      setSearchError('Search failed. See console for details.');
      setSearchLoading(false);
    }
  };

  // Toggle selection for relevance feedback
  const toggleSelectDoc = (docId) => {
    const s = new Set(selectedDocs);
    if (s.has(docId)) s.delete(docId);
    else s.add(docId);
    setSelectedDocs(s);
  };

  // Send selected docs to backend to perform Rocchio expansion and re-run search
  const refineWithSelected = async () => {
    if (selectedDocs.size === 0) return;
    setRefineLoading(true);
    setSearchError(null);

    const payload = {
      q: query,
      selected_doc_ids: Array.from(selectedDocs),
      k: k,
      method: method,
      alpha: 1.0,
      beta: 0.75,
      top_terms: 12
    };

    try {
      const data = await feedbackApi(API_URL, payload);
      setRefineLoading(false);
      setResults(Array.isArray(data.results) ? data.results : []);
      setSelectedDocs(new Set());
      if (data.expanded_query_terms) {
        console.info('Expanded query terms:', data.expanded_query_terms.join(', '));
      }
    } catch (err) {
      console.error('Refine error:', err);
      setSearchError('Refinement failed. See console for details.');
      setRefineLoading(false);
    }
  };

  // Topic selection handler (called when user clicks a topic in TopicsPanel)
  const handleTopicSelect = async (topicId) => {
    // Clear results if topicId is null
    if (topicId === null || topicId === undefined) {
      setResults([]);
      setSearchError(null);
      setSelectedDocs(new Set());
      return;
    }
    
    // fetch docs for this topic and show them as results
    setSearchLoading(true);
    setSearchError(null);
    setResults([]);
    setSelectedDocs(new Set());

    try {
      const resp = await getTopicDocs(API_URL, topicId, 50, 0);
      // map topic docs into same result shape used by ResultsList
      const docs = (resp.docs || []).map(d => ({
        doc_id: d.doc_id,
        title: d.title,
        summary: d.summary,
        snippet: (d.summary || '').slice(0, 300),
        score: 0.0,
        published: d.published,
        authors: d.authors,
        link: d.link
      }));
      setResults(docs);
      setSearchLoading(false);
    } catch (err) {
      console.error('Topic docs error:', err);
      setSearchError('Failed to load topic documents. See console.');
      setSearchLoading(false);
    }
  };

  // Cluster selection handler (called when user clicks a cluster in TopicsPanel)
  const handleClusterSelect = async (clusterId) => {
    if (clusterId === null || clusterId === undefined) {
      setResults([]);
      setSearchError(null);
      setSelectedDocs(new Set());
      return;
    }
    setSearchLoading(true);
    setSearchError(null);
    setResults([]);
    setSelectedDocs(new Set());

    try {
      const resp = await getClusterDocs(API_URL, clusterId, 50, 0);
      const docs = (resp.docs || []).map(d => ({
        doc_id: d.doc_id,
        title: d.title,
        summary: d.summary,
        snippet: (d.summary || '').slice(0, 300),
        score: 0.0,
        published: d.published,
        authors: d.authors,
        link: d.link
      }));
      setResults(docs);
      setSearchLoading(false);
    } catch (err) {
      console.error('Cluster docs error:', err);
      setSearchError('Failed to load cluster documents. See console.');
      setSearchLoading(false);
    }
  };

  const clearAll = () => {
    setQuery('');
    setResults([]);
    setSearchError(null);
    setSelectedDocs(new Set());
    setPredictedCluster(null);
    setPredictedTopic(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <Header
        fetchPapers={fetchPapers}
        fetchLoading={fetchLoading}
        buildEmbeddings={buildEmbeddings}
        buildLoading={buildLoading}
        buildStatus={buildStatus}
        buildLdaTopics={buildLdaTopics}
        buildLdaLoading={buildLdaLoading}
        buildLdaStatus={buildLdaStatus}
        buildClusters={() => buildClusters({ method: 'hdbscan', min_cluster_size: 8 })}
        buildClustersLoading={buildClustersLoading}
        buildClustersStatus={buildClustersStatus}
      />

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
        <div className="lg:col-span-1 space-y-6">
          <SearchForm
            query={query}
            setQuery={setQuery}
            method={method}
            setMethod={setMethod}
            k={k}
            setK={setK}
            startDate={startDate}
            setStartDate={setStartDate}
            endDate={endDate}
            setEndDate={setEndDate}
            author={author}
            setAuthor={setAuthor}
            hybridWeights={hybridWeights}
            setHybridWeights={setHybridWeights}
            runSearch={runSearch}
            searchLoading={searchLoading}
            fetchStatus={fetchStatus}
            fetchError={fetchError}
            clearAll={clearAll}
            clusterMode={clusterMode}
            setClusterMode={setClusterMode}
          />

          {/* Topics/Clusters panel shown under the search form (same page) */}
          <TopicsPanel apiUrl={API_URL} onTopicSelect={handleTopicSelect} onClusterSelect={handleClusterSelect} />
        </div>

        <section className="lg:col-span-2">
          {predictedTopic && (
            <div className="p-3 mb-3 rounded bg-yellow-50 text-sm border border-yellow-100">
              Likely topic: <strong>{predictedTopic.top_terms.slice(0,6).join(', ')}</strong>
              {predictedCluster && (
                <>
                  {' '}— Cluster <strong>{predictedCluster.cluster_id}</strong> (size {predictedCluster.size}){predictedClusterScore && ` — score ${predictedClusterScore.toFixed(3)}`}
                  <button onClick={() => handleTopicSelect(predictedTopic.topic_id)} className="ml-3 px-2 py-1 text-xs bg-indigo-600 text-white rounded">Show topic docs</button>
                </>
              )}
            </div>
          )}

          <ResultsList
            results={results}
            selectedDocs={selectedDocs}
            toggleSelectDoc={toggleSelectDoc}
            openLink={(link) => window.open(link, '_blank', 'noopener,noreferrer')}
            searchLoading={searchLoading}
          />

          <FeedbackBar
            selectedCount={selectedDocs.size}
            onRefine={refineWithSelected}
            onClear={() => setSelectedDocs(new Set())}
            refineLoading={refineLoading}
          />
        </section>
      </main>

    </div>
  );
}