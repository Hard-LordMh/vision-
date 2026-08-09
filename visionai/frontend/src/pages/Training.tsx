import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { BrainCircuit, Play, Loader2, Gauge } from 'lucide-react';

import { getApiUrl, getWsUrl } from '../config';

export default function Training() {
  const [datasets, setDatasets] = useState([]);
  const [projectMode, setProjectMode] = useState<'vision' | 'placement'>(() => {
    return (localStorage.getItem('visionai_project_mode') as 'vision' | 'placement') || 'vision';
  });
  
  const [config, setConfig] = useState({
    dataset_id: '',
    model_architecture: 'MobileNetV3',
    epochs: 10,
    batch_size: 16,
    learning_rate: 0.001
  });
  
  const [activeRunId, setActiveRunId] = useState<number | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<any>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const api = getApiUrl();
    axios.get(`${api}/datasets`).then(res => setDatasets(res.data));
    
    // Check if there's an ongoing run
    axios.get(`${api}/training`).then(res => {
      const active = res.data.find((r: any) => r.status === 'TRAINING' || r.status === 'READY');
      if (active) {
        connectWebSocket(active.id);
      }
    });

    const handleModeChange = () => {
      const mode = (localStorage.getItem('visionai_project_mode') as 'vision' | 'placement') || 'vision';
      setProjectMode(mode);
      setConfig(prev => ({
        ...prev,
        model_architecture: mode === 'placement' ? 'ANN-3-Layer' : 'MobileNetV3',
        dataset_id: ''
      }));
    };
    window.addEventListener('projectModeChanged', handleModeChange);
    
    return () => {
      if (wsRef.current) wsRef.current.close();
      window.removeEventListener('projectModeChanged', handleModeChange);
    };
  }, []);

  const connectWebSocket = (runId: number) => {
    setActiveRunId(runId);
    if (wsRef.current) wsRef.current.close();
    
    const ws = new WebSocket(getWsUrl(runId));
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLiveMetrics(data);
      if (data.status === 'COMPLETED' || data.status === 'FAILED') {
        ws.close();
        setActiveRunId(null);
      }
    };
    wsRef.current = ws;
  };

  const startTraining = async () => {
    if (!config.dataset_id) return alert("Select a dataset first");
    
    try {
      const res = await axios.post(`${getApiUrl()}/training/start`, {
        ...config,
        dataset_id: parseInt(config.dataset_id)
      });
      connectWebSocket(res.data.id);
    } catch (e) {
      console.error(e);
      alert("Failed to start training");
    }
  };

  const progress = liveMetrics ? (liveMetrics.epoch / liveMetrics.epochs) * 100 : 0;

  const filteredDatasets = datasets.filter((ds: any) => {
    const isPlacement = projectMode === 'placement';
    const hasPlacementName = ds.name.toLowerCase().includes('placement');
    return isPlacement ? hasPlacementName : !hasPlacementName;
  });

  return (
    <div className="space-y-10 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-12">
      <header className="relative">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-2 uppercase">
          Neural Core<span className="text-zinc-500">.</span>
        </h1>
        <p className="text-zinc-400 text-lg tracking-wide max-w-xl font-light">
          Configure model parameters and monitor training telemetry.
        </p>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        {/* Configuration Panel */}
        <div className="xl:col-span-1 space-y-6">
          <div className="glass-panel rounded-3xl p-8 shadow-2xl relative overflow-hidden group transition-all duration-300">
            <div className="absolute top-0 right-0 w-48 h-48 bg-white/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />
            <h2 className="text-xl font-bold mb-8 flex items-center gap-3 uppercase tracking-widest text-zinc-100">
              <BrainCircuit className="w-6 h-6 text-white" />
              Parameters
            </h2>
            
            <div className="space-y-6 relative z-10">
              <div>
                <label className="block text-xs font-bold text-zinc-400 mb-2 tracking-[0.1em] uppercase">Dataset Input</label>
                <select 
                  value={config.dataset_id} 
                  onChange={e => setConfig({...config, dataset_id: e.target.value})}
                  disabled={activeRunId !== null}
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-white/30 transition-colors appearance-none cursor-pointer"
                >
                  <option value="">-- SELECT DATASET --</option>
                  {filteredDatasets.map((ds: any) => (
                    <option key={ds.id} value={ds.id}>{ds.name} ({ds.class_count} {projectMode === 'vision' ? 'classes' : 'outcomes'})</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-bold text-zinc-400 mb-2 tracking-[0.1em] uppercase">Network Architecture</label>
                <select 
                  value={config.model_architecture} 
                  onChange={e => setConfig({...config, model_architecture: e.target.value})}
                  disabled={activeRunId !== null}
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-white/30 transition-colors appearance-none cursor-pointer"
                >
                  {projectMode === 'vision' ? (
                    <>
                      <option value="MobileNetV3">MobileNetV3 (Fast, CPU Friendly)</option>
                      <option value="EfficientNet-B0">EfficientNet-B0 (Accurate)</option>
                      <option value="ResNet50">ResNet50 (Deep, Powerful CNN)</option>
                      <option value="ConvNeXt-Tiny">ConvNeXt-Tiny (Modern, Robust CNN)</option>
                      <option value="ViT-B/16">ViT-B/16 (State-of-the-Art Vision Transformer)</option>
                    </>
                  ) : (
                    <>
                      <option value="ANN-3-Layer">ANN-3-Layer (Dense: 64 ➔ 32 ➔ 2)</option>
                      <option value="ANN-5-Layer">ANN-5-Layer (Dense: 128 ➔ 64 ➔ 32 ➔ 16 ➔ 2)</option>
                    </>
                  )}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-bold text-zinc-400 mb-2 tracking-[0.1em] uppercase">Epochs</label>
                  <input 
                    type="number" min="1" max="50"
                    value={config.epochs} 
                    onChange={e => setConfig({...config, epochs: parseInt(e.target.value)})}
                    disabled={activeRunId !== null}
                    className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-white/30 transition-colors"
                  />
                </div>
                <div>
                  <label className="block text-xs font-bold text-zinc-400 mb-2 tracking-[0.1em] uppercase">Batch Size</label>
                  <select 
                    value={config.batch_size} 
                    onChange={e => setConfig({...config, batch_size: parseInt(e.target.value)})}
                    disabled={activeRunId !== null}
                    className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-white/30 transition-colors appearance-none"
                  >
                    <option value="8">8</option>
                    <option value="16">16</option>
                    <option value="32">32</option>
                  </select>
                </div>
              </div>

              <button 
                onClick={startTraining}
                disabled={activeRunId !== null || !config.dataset_id}
                className="w-full mt-8 flex items-center justify-center gap-3 bg-white hover:bg-zinc-200 disabled:bg-white/5 disabled:text-zinc-500 text-black font-extrabold py-4 px-4 rounded-xl transition-all shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.2)] disabled:shadow-none tracking-[0.1em] uppercase text-sm"
              >
                {activeRunId ? (
                  <><Loader2 className="w-5 h-5 animate-spin" /> Training Online</>
                ) : (
                  <><Play className="w-5 h-5 fill-current" /> Ignite Engine</>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Live Metrics Panel */}
        <div className="xl:col-span-2">
          <div className="glass-panel rounded-3xl p-8 shadow-2xl h-full flex flex-col relative overflow-hidden group transition-all duration-300">
            <div className="absolute bottom-0 left-0 w-64 h-64 bg-white/5 rounded-full blur-3xl translate-y-1/2 -translate-x-1/4 pointer-events-none" />
            
            <div className="flex justify-between items-center mb-10 relative z-10">
              <h2 className="text-xl font-bold flex items-center gap-3 uppercase tracking-widest text-zinc-100">
                <Gauge className="w-6 h-6 text-white" />
                Live Telemetry
              </h2>
              {liveMetrics?.status === 'TRAINING' && (
                <span className="flex items-center gap-3 text-white text-xs font-bold tracking-[0.2em] bg-white/10 border border-white/20 px-4 py-2 rounded-full shadow-[0_0_15px_rgba(255,255,255,0.05)]">
                  <div className="w-2 h-2 bg-white rounded-full animate-ping" />
                  RUNTIME {activeRunId}
                </span>
              )}
            </div>

            {liveMetrics ? (
              <div className="flex-1 space-y-12 relative z-10">
                <div className="space-y-4">
                  <div className="flex justify-between text-xs font-bold tracking-widest uppercase">
                    <span className="text-zinc-400">Compilation Progress</span>
                    <span className="text-white text-shadow-[0_0_10px_rgba(255,255,255,0.2)]">Epoch {liveMetrics.epoch} / {liveMetrics.epochs}</span>
                  </div>
                  <div className="h-4 w-full bg-black/60 rounded-full overflow-hidden border border-white/10 shadow-inner">
                    <div 
                      className="h-full bg-zinc-400 transition-all duration-1000 ease-out relative shadow-[0_0_10px_rgba(255,255,255,0.3)]"
                      style={{ width: `${progress}%` }}
                    >
                      <div className="absolute top-0 right-0 bottom-0 left-0 bg-[linear-gradient(45deg,transparent_25%,rgba(255,255,255,.3)_25%,rgba(255,255,255,.3)_50%,transparent_50%,transparent_75%,rgba(255,255,255,.3)_75%,rgba(255,255,255,.3)_100%)] bg-[length:20px_20px] animate-[progress_1s_linear_infinite]" />
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                  <MetricCard title="Train Loss" value={liveMetrics.train_loss?.toFixed(4) || '--'} />
                  <MetricCard title="Train Acc" value={liveMetrics.train_accuracy ? `${(liveMetrics.train_accuracy * 100).toFixed(2)}%` : '--'} />
                  <MetricCard title="Val Loss" value={liveMetrics.val_loss?.toFixed(4) || '--'} highlight />
                  <MetricCard title="Val Acc" value={liveMetrics.val_accuracy ? `${(liveMetrics.val_accuracy * 100).toFixed(2)}%` : '--'} highlight />
                </div>
                
                {liveMetrics.status === 'COMPLETED' && (
                  <div className="mt-8 p-6 bg-white/5 border border-white/20 rounded-2xl text-white text-center font-bold tracking-widest uppercase shadow-[0_0_20px_rgba(255,255,255,0.05)]">
                    Compilation Successful. Model logged to registry.
                  </div>
                )}
                {liveMetrics.status === 'FAILED' && (
                  <div className="mt-8 p-6 bg-red-900/20 border border-red-500/30 rounded-2xl text-red-400 text-center font-bold tracking-widest uppercase shadow-[0_0_20px_rgba(239,68,68,0.1)]">
                    Critical Error in compilation pipeline.
                  </div>
                )}
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-500 relative z-10">
                <BrainCircuit className="w-24 h-24 mb-6 opacity-10" />
                <p className="font-light tracking-[0.2em] uppercase text-sm text-zinc-600">System standing by for configuration...</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, highlight = false }: { title: string, value: string | number, highlight?: boolean }) {
  return (
    <div className={`p-6 rounded-2xl border transition-all duration-300 ${highlight ? 'bg-white/5 border-white/20 shadow-[0_0_15px_rgba(255,255,255,0.05)]' : 'bg-black/30 border-white/5'}`}>
      <h3 className="text-xs font-bold text-zinc-500 mb-2 tracking-[0.1em] uppercase">{title}</h3>
      <p className={`text-3xl font-extrabold ${highlight ? 'text-white text-shadow-[0_0_10px_rgba(255,255,255,0.2)]' : 'text-zinc-300'}`}>{value}</p>
    </div>
  );
}

