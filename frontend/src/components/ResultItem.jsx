'use client'
export default function ResultItem({ r, idx, isSelected, onToggle, onOpenLink }) {
  const link = r.link || r.url || r.id || r.arxiv_link || null;
  return (
    <li className="group block bg-white border border-gray-100 hover:shadow-md transition rounded-lg p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="pt-1">
            <input
              type="checkbox"
              checked={isSelected}
              onChange={onToggle}
              className="h-4 w-4 text-indigo-600 border-gray-300 rounded"
              title="Select as relevant for feedback"
            />
          </div>

          <div className="flex-1">
            <div className="flex items-center gap-3">
              <a
                href={link || '#'}
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-900 font-medium text-lg hover:underline"
                onClick={(e) => { e.stopPropagation(); }}
              >
                {r.title || 'Untitled'}
              </a>

              {/* Badges: cluster and topic */}
              <div className="ml-2 flex items-center gap-2">
                {r.cluster_id !== undefined && r.cluster_id !== null && (
                  <span className="text-xs px-2 py-0.5 rounded bg-rose-100 text-rose-800">Cluster {r.cluster_id}</span>
                )}
                {r.topic_lda_id !== undefined && r.topic_lda_id !== null && (
                  <span className="text-xs px-2 py-0.5 rounded bg-yellow-100 text-yellow-800">Topic {r.topic_lda_id}</span>
                )}
              </div>
            </div>

            <div className="mt-2 text-sm text-slate-600">
              <span className="mr-3"><strong>Authors:</strong> {Array.isArray(r.authors) ? r.authors.join(', ') : r.authors || 'N/A'}</span>
              <span><strong>Published:</strong> {r.published || 'N/A'}</span>
            </div>

            <p className="mt-3 text-slate-700 line-clamp-3">
              {r.snippet || (r.summary ? (r.summary.length > 300 ? r.summary.slice(0, 300) + '...' : r.summary) : '')}
            </p>
          </div>
        </div>

        <div className="w-36  text-right">
          <div className="text-indigo-600 font-semibold">{(r.score || 0).toFixed(4)}</div>
          <div className="text-xs text-slate-400 mt-2">rank #{idx + 1}</div>
          {link && (
            <div className="mt-3">
              <a href={link} target="_blank" rel="noopener noreferrer"
                 className="inline-flex items-center gap-2 px-3 py-1 rounded-md bg-indigo-50 text-indigo-700 text-xs hover:bg-indigo-100"
                 onClick={(e) => e.stopPropagation()}
              >
                View paper
              </a>
            </div>
          )}
        </div>
      </div>
    </li>
  );
}