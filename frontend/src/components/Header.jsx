'use client'
export default function Header({
  fetchPapers,
  fetchLoading,
  buildEmbeddings,
  buildLoading,
  buildStatus,
  buildLdaTopics,
  buildLdaLoading,
  buildLdaStatus,
  buildClusters,
  buildClustersLoading,
  buildClustersStatus
}) {
  return (
    <header className="max-w-6xl mx-auto">
      <div className="relative bg-white rounded-lg shadow-md p-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">Research Papers IR System</h1>
          <p className="text-sm text-slate-500 mt-1">Search arXiv CS papers with BM25, TF‑IDF, Cosine, Semantic or a hybrid ranking.</p>
        </div>

        <div className="absolute right-6 top-6 flex items-center gap-3">
          <button
            onClick={fetchPapers}
            disabled={fetchLoading}
            className={`inline-flex items-center gap-3 px-4 py-2 rounded-md text-sm font-medium transition
              ${fetchLoading ? 'bg-blue-300 text-white cursor-not-allowed' : 'bg-blue-600 text-white hover:bg-blue-700'}`}
          >
            {fetchLoading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2v4a6 6 0 0 1 0 12v4c4.418 0 8-3.582 8-8s-3.582-8-8-8z"/></svg>
                Fetching...
              </>
            ) : (
              'Fetch Papers'
            )}
          </button>

          <button
            onClick={buildEmbeddings}
            disabled={buildLoading}
            className={`inline-flex items-center gap-3 px-4 py-2 rounded-md text-sm font-medium transition
              ${buildLoading ? 'bg-emerald-300 text-white cursor-not-allowed' : 'bg-emerald-600 text-white hover:bg-emerald-700'}`}
          >
            {buildLoading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2v4a6 6 0 0 1 0 12v4c4.418 0 8-3.582 8-8s-3.582-8-8-8z"/></svg>
                Building...
              </>
            ) : (
              'Build Embeddings'
            )}
          </button>

          <button
            onClick={() => buildLdaTopics({ num_topics: 12, passes: 10 })}
            disabled={buildLdaLoading}
            className={`inline-flex items-center gap-3 px-4 py-2 rounded-md text-sm font-medium transition
              ${buildLdaLoading ? 'bg-yellow-300 text-white cursor-not-allowed' : 'bg-yellow-600 text-white hover:bg-yellow-700'}`}
          >
            {buildLdaLoading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2v4a6 6 0 0 1 0 12v4c4.418 0 8-3.582 8-8s-3.582-8-8-8z"/></svg>
                Building Topics...
              </>
            ) : (
              'Build Topics'
            )}
          </button>

          <button
            onClick={buildClusters}
            disabled={buildClustersLoading}
            className={`inline-flex items-center gap-3 px-4 py-2 rounded-md text-sm font-medium transition
              ${buildClustersLoading ? 'bg-rose-300 text-white cursor-not-allowed' : 'bg-rose-600 text-white hover:bg-rose-700'}`}
          >
            {buildClustersLoading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24"><path fill="currentColor" d="M12 2v4a6 6 0 0 1 0 12v4c4.418 0 8-3.582 8-8s-3.582-8-8-8z"/></svg>
                Clustering...
              </>
            ) : (
              'Build Clusters'
            )}
          </button>
        </div>
      </div>

      {/* optional status lines under header */}
      {buildStatus && buildStatus.success && (
        <div className="max-w-6xl mx-auto mt-3 p-3 rounded-md bg-emerald-50 text-emerald-700 text-sm border border-emerald-100">
          Embeddings built: {buildStatus.info?.total ?? 'N/A'} docs.
        </div>
      )}

      {buildLdaStatus && buildLdaStatus.success && (
        <div className="max-w-6xl mx-auto mt-3 p-3 rounded-md bg-yellow-50 text-yellow-800 text-sm border border-yellow-100">
          Topics built: {buildLdaStatus.info?.num_topics ?? 'N/A'} topics for {buildLdaStatus.info?.num_papers ?? 'N/A'} docs.
        </div>
      )}

      {buildClustersStatus && buildClustersStatus.success && (
        <div className="max-w-6xl mx-auto mt-3 p-3 rounded-md bg-rose-50 text-rose-800 text-sm border border-rose-100">
          Clusters built: {buildClustersStatus.info?.num_clusters ?? 'N/A'} clusters for {buildClustersStatus.info?.num_docs ?? 'N/A'} docs.
        </div>
      )}
    </header>
  );
}