export async function searchApi(apiUrl, params = {}) {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([k,v]) => {
    if (v !== undefined && v !== null && v !== '') searchParams.set(k, v);
  });

  // If method is semantic, call the dedicated semantic endpoint
  if (params.method === 'semantic') {
    const url = `${apiUrl}/search/semantic?${searchParams.toString()}`;
    const res = await fetch(url);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Server ${res.status}: ${text}`);
    }
    return res.json();
  }

  const url = `${apiUrl}/search?${searchParams.toString()}`;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  return res.json();
}

export async function fetchPapersApi(apiUrl) {
  const res = await fetch(`${apiUrl}/papers`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  return res.json();
}
export async function buildEmbeddingsApi(apiUrl) {
  const res = await fetch(`${apiUrl}/embeddings/build`, { method: 'POST' });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  return res.json();
}

export async function buildLdaTopicsApi(apiUrl, params = {}) {
  // POST /topics/lda/build with query params
  const url = new URL(`${apiUrl}/topics/lda/build`);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v));
  });
  const res = await fetch(url.toString(), { method: 'POST' });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  return res.json();
}

export async function feedbackApi(apiUrl, payload) {
  const res = await fetch(`${apiUrl}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getLdaTopics(apiUrl) {
  const res = await fetch(`${apiUrl}/topics/lda`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  return res.json();
} 

export async function getTopicDocs(apiUrl, topicId, limit = 20, skip = 0) {
  const url = `${apiUrl}/topics/lda/${topicId}/docs?limit=${limit}&skip=${skip}`;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getUmapPoints(apiUrl, limit = 1000) {
  const url = `${apiUrl}/topics/lda/umap${limit ? `?limit=${limit}` : ""}`;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  return res.json();
}


export async function buildClustersApi(apiUrl, params = {}) {
  console.log("Building clusters with params:", params);
  const url = new URL(`${apiUrl}/clusters/build`);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v));
  });
  const res = await fetch(url.toString(), { method: 'POST' });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  console.log("Build clusters response received");
  return res.json();
}  

export async function searchClusteredApi(apiUrl, params = {}) {
  const url = new URL(`${apiUrl}/search/clustered`);
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v));
  });
  const res = await fetch(url.toString());
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  return res.json();
}




export async function getClusters(apiUrl) {
  const res = await fetch(`${apiUrl}/clusters`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getClustersUmap(apiUrl, limit = 400) {
  const url = `${apiUrl}/clusters/umap${limit ? `?limit=${limit}` : ''}`;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  return res.json();
}

export async function getClusterDocs(apiUrl, clusterId, limit = 50, skip = 0) {
  const url = `${apiUrl}/clusters/${clusterId}/docs?limit=${limit}&skip=${skip}`;
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Server ${res.status}: ${text}`);
  }
  return res.json();
}