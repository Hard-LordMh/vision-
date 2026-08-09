export const getApiUrl = () => {
  const stored = localStorage.getItem('visionai_api_url');
  if (stored) return stored;
  
  // Resolve API hostname based on current page's host (fixes mobile/LAN access)
  const hostname = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
  return `http://${hostname}:8000/api`;
};

export const getWsUrl = (runId: number | string) => {
  const apiUrl = getApiUrl();
  try {
    const url = new URL(apiUrl);
    const protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    // Remove trailing slash and map to WebSocket path
    const cleanPath = url.pathname.replace(/\/$/, '');
    return `${protocol}//${url.host}${cleanPath}/ws/training/${runId}`;
  } catch (e) {
    return `ws://localhost:8000/api/ws/training/${runId}`;
  }
};

export const getDefaultThreshold = () => {
  const stored = localStorage.getItem('visionai_default_threshold');
  return stored ? parseFloat(stored) : 0.60;
};
