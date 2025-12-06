'use client'
import ResultItem from './ResultItem';

export default function ResultsList({ results, selectedDocs, toggleSelectDoc, openLink, searchLoading }) {
  return (
    <div className="bg-white rounded-lg shadow-sm p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-slate-800">Results <span className="text-sm text-slate-500">({results.length})</span></h2>
      </div>

      {searchLoading && (
        <div className="space-y-2">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="animate-pulse h-20 rounded-md bg-gray-100" />
          ))}
        </div>
      )}

      {!searchLoading && results.length === 0 && (
        <div className="py-8 text-center text-slate-500">No results yet — run a search or fetch papers.</div>
      )}

      <ul className="space-y-4">
        {results.map((r, idx) => (
          <ResultItem
            key={r.doc_id ?? `${idx}-${(r.title || '').slice(0,10)}`}
            r={r}
            idx={idx}
            isSelected={selectedDocs.has(r.doc_id)}
            onToggle={() => toggleSelectDoc(r.doc_id)}
            onOpenLink={() => openLink(r.link || r.url || r.id || r.arxiv_link)}
          />
        ))}
      </ul>
    </div>
  );
}