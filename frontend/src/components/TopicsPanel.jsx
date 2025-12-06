'use client'
import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { getLdaTopics, getUmapPoints, getClustersUmap, getClusters } from '@/utils/api';

// Plotly relies on browser globals; load only on client.
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

export default function TopicsPanel({ apiUrl, onTopicSelect, onClusterSelect }) {
  const [mode, setMode] = useState('topics'); // 'topics' or 'clusters'
  const [topics, setTopics] = useState([]);
  const [topicPoints, setTopicPoints] = useState([]);
  const [clusterPoints, setClusterPoints] = useState([]);
  const [clustersMeta, setClustersMeta] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedId, setSelectedId] = useState(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        // Parallelize all API calls instead of sequential
        const [t, tp, cm, cp] = await Promise.all([
          getLdaTopics(apiUrl),
          getUmapPoints(apiUrl, 400),
          getClusters(apiUrl),
          getClustersUmap(apiUrl, 400)
        ]);
        
        setTopics(t.topics || []);
        setTopicPoints(tp.points || []);
        setClustersMeta(cm.clusters || []);
        setClusterPoints(cp.points || []);
      } catch (err) {
        console.error('TopicsPanel load error', err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [apiUrl]);

  const handleSelect = (id) => {
    setSelectedId(id);
    if (mode === 'topics') {
      if (onTopicSelect) onTopicSelect(id);
    } else {
      if (onClusterSelect) onClusterSelect(id);
    }
  };

  const renderPlot = () => {
    const source = mode === 'topics' ? topicPoints : clusterPoints;
    const keyField = mode === 'topics' ? 'topic_id' : 'cluster_id';
    if (!source || source.length === 0) return <div className="text-xs text-slate-400">No visualization data available</div>;

    const grouped = {};
    for (const p of source) {
      const k = p[keyField] ?? -1;
      if (!grouped[k]) grouped[k] = { x: [], y: [], text: [], ids: [] };
      grouped[k].x.push(p.x);
      grouped[k].y.push(p.y);
      grouped[k].text.push(`${p.title}\n${p.snippet}`);
      grouped[k].ids.push(p.doc_id);
    }

    const colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];

    const traces = Object.keys(grouped).map((k, idx) => ({
      x: grouped[k].x,
      y: grouped[k].y,
      mode: 'markers',
      type: 'scattergl',
      name: `${mode === 'topics' ? 'Topic ' : 'Cluster '}${k}`,
      marker: { size: 6, color: colors[idx % colors.length], opacity: selectedId === null ? 0.85 : (parseInt(k) === selectedId ? 1.0 : 0.12) },
      text: grouped[k].text,
      customdata: grouped[k].ids,
      hoverinfo: 'text',
      showlegend: false
    }));

    return (
      <Plot
        data={traces}
        layout={{ width: 340, height: 280, margin: { l: 20, r: 20, t: 30, b: 20 } }}
        onClick={(e) => {
          if (e && e.points && e.points.length) {
            const pt = e.points[0];
            // parse label or use customdata
            if (pt.fullData && pt.fullData.name) {
              const label = pt.fullData.name;
              const id = parseInt(label.split(' ').pop());
              handleSelect(id);
            } else if (pt.customdata) {
              // fallback: get doc id and lookup its cluster/topic via API if needed
              const docId = pt.customdata;
              // optionally open that doc or show details
            }
          }
        }}
      />
    );
  };

  return (
    <div className="bg-white rounded-lg shadow-sm p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold">{mode === 'topics' ? 'Topics' : 'Clusters'}</h3>
        <div className="flex gap-1">
          <button className={`px-2 py-1 text-xs ${mode==='topics' ? 'bg-indigo-50 rounded' : 'bg-white'}`} onClick={() => setMode('topics')}>Topics</button>
          <button className={`px-2 py-1 text-xs ${mode==='clusters' ? 'bg-indigo-50 rounded' : 'bg-white'}`} onClick={() => setMode('clusters')}>Clusters</button>
        </div>
      </div>

      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2  max-h-48 overflow-y-auto">
          {mode === 'topics'
            ? topics.slice(0, topics.length).map(t => (
                <button key={t.topic_id} onClick={() => handleSelect(t.topic_id)} className={`text-xs p-2 rounded border text-left ${selectedId === t.topic_id ? 'bg-indigo-50' : 'bg-white'}`}>
                  <div className="font-medium">Topic {t.topic_id} ({t.size})</div>
                  <div className="text-slate-500 mt-1">{t.top_terms.slice(0,4).join(', ')}</div>
                </button>
              ))
            : clustersMeta.slice(0, clustersMeta.length).map(c => (
                <button key={c.cluster_id} onClick={() => handleSelect(c.cluster_id)} className={`text-xs p-2 rounded border text-left ${selectedId === c.cluster_id ? 'bg-indigo-50' : 'bg-white'}`}>
                  <div className="font-medium">Cluster {c.cluster_id} ({c.size})</div>
                  <div className="text-slate-500 mt-1">Top topic: {c.top_topic ?? 'N/A'}</div>
                </button>
              ))
          }
        </div>

        <div className="mt-2">
          {loading ? <div className="text-xs text-slate-400">loading...</div> : renderPlot()}
        </div>

        <div className="flex gap-2">
          <button onClick={() => { setSelectedId(null); if (mode === 'topics' && onTopicSelect) onTopicSelect(null); if (mode === 'clusters' && onClusterSelect) onClusterSelect(null); }}
            className="px-3 py-1 text-sm rounded border bg-white hover:bg-gray-50">Clear</button>
          <button onClick={() => { if (mode === 'topics' && selectedId !== null && onTopicSelect) onTopicSelect(selectedId); if (mode === 'clusters' && selectedId !== null && onClusterSelect) onClusterSelect(selectedId); }}
            className="px-3 py-1 text-sm rounded bg-indigo-600 text-white">Show</button>
        </div>
      </div>
    </div>
  );
}