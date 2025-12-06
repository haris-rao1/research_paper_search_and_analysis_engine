'use client'
export default function SearchForm({
  query, setQuery,
  method, setMethod,
  k, setK,
  startDate, setStartDate,
  endDate, setEndDate,
  author, setAuthor,
  hybridWeights, setHybridWeights,
  runSearch, searchLoading,
  fetchStatus, fetchError, clearAll,
  clusterMode, setClusterMode
}) {
  return (
    <section className="lg:col-span-1">
      <div className="bg-white rounded-lg shadow-sm p-6">
        <form onSubmit={runSearch} className="space-y-4">
          <label className="block">
            <span className="text-sm font-medium text-slate-700">Query</span>
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter search terms (e.g. deep learning, transformer models)"
              className="mt-1 block w-full rounded-md border-gray-200 shadow-sm focus:ring-blue-500 focus:border-blue-500"
            />
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-sm font-medium text-slate-700">Method</span>
              <select value={method} onChange={(e) => setMethod(e.target.value)} className="mt-1 block w-full rounded-md border-gray-200">
                <option value="bm25">BM25</option>
                <option value="tfidf">TF‑IDF</option>
                <option value="cosine">Cosine</option>
                <option value="semantic">Semantic</option>
                <option value="hybrid">Hybrid</option>
              </select>
            </label>

            <label className="block">
              <span className="text-sm font-medium text-slate-700">Results (k)</span>
              <input type="number" min="1" max="100" value={k} onChange={(e) => setK(Number(e.target.value))}
                className="mt-1 block w-full rounded-md border-gray-200" />
            </label>
          </div>

          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={clusterMode} onChange={(e) => setClusterMode(e.target.checked)} className="h-4 w-4" />
              <span>Cluster-first search</span>
            </label>
            <span className="text-xs text-slate-400 ml-2">(Find cluster → topic → rank inside cluster)</span>
          </div>

          <label className="block">
            <span className="text-sm font-medium text-slate-700">Author (optional)</span>
            <input value={author} onChange={(e) => setAuthor(e.target.value)} placeholder="Filter by author"
              className="mt-1 block w-full rounded-md border-gray-200" />
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-sm font-medium text-slate-700">Start date</span>
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="mt-1 block w-full rounded-md border-gray-200" />
            </label>

            <label className="block">
              <span className="text-sm font-medium text-slate-700">End date</span>
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="mt-1 block w-full rounded-md border-gray-200" />
            </label>
          </div>

          {method === 'hybrid' && (
            <label className="block">
              <span className="text-sm font-medium text-slate-700">Hybrid weights (bm25,tfidf,cosine)</span>
              <input value={hybridWeights} onChange={(e) => setHybridWeights(e.target.value)}
                placeholder="0.33,0.33,0.34" className="mt-1 block w-full rounded-md border-gray-200" />
              <p className="text-xs text-slate-400 mt-1">Three comma-separated numbers summing to 1 (optional).</p>
            </label>
          )}

          <div className="flex gap-3">
            <button type="submit" disabled={searchLoading}
              className={`inline-flex items-center gap-2 px-4 py-2 rounded-md font-medium text-white ${searchLoading ? 'bg-indigo-300 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'}`}>
              {searchLoading ? 'Searching...' : 'Search'}
            </button>

            <button type="button" onClick={clearAll}
              className="px-4 py-2 rounded-md border border-gray-200 bg-white text-sm hover:bg-gray-50">
              Clear
            </button>
          </div>

          {fetchError && <p className="text-sm text-red-600">{fetchError}</p>}
          {fetchStatus && fetchStatus.success && (
            <div className="mt-3 p-3 rounded-md bg-green-50 text-sm text-green-700 border border-green-100">
              <div className="font-medium mb-2">
                ✓ Fetched {fetchStatus.count} papers: {fetchStatus.inserted} new, {fetchStatus.skipped || 0} skipped
              </div>

              {fetchStatus.indexStats && (
                <div className="text-xs text-green-600 mb-2">
                  Index: {fetchStatus.indexStats.unique_terms} unique terms |
                  {fetchStatus.indexStats.total_docs} docs |
                  Avg length: {fetchStatus.indexStats.avg_doc_length}
                </div>
              )}

            </div>
          )}
        </form>
      </div>
    </section>
  );
}