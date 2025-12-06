'use client'
export default function FeedbackBar({ selectedCount, onRefine, onClear, refineLoading }) {
  return (
    <div className="mt-4 flex items-center justify-between gap-4">
      <div className="text-sm text-slate-500">
        Select up to 5 relevant documents from the list, then click <span className="font-semibold">Refine search</span> to apply relevance feedback.
        <div className="text-xs text-slate-400 mt-1">Selected: <span className="font-medium text-slate-700">{selectedCount}</span></div>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={onRefine}
          disabled={selectedCount === 0 || refineLoading}
          className={`inline-flex items-center gap-2 px-4 py-2 rounded-md font-medium text-white ${refineLoading ? 'bg-emerald-300 cursor-not-allowed' : 'bg-emerald-600 hover:bg-emerald-700'}`}
        >
          {refineLoading ? 'Refining...' : 'Refine search'}
        </button>

        <button
          onClick={onClear}
          className="px-3 py-2 rounded-md border border-gray-200 bg-white text-sm hover:bg-gray-50"
        >
          Clear selection
        </button>
      </div>
    </div>
  );
}