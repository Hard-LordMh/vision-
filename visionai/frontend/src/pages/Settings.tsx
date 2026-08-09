import { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Save, Cpu, RefreshCw, Layers, 
  Link as LinkIcon, Sliders, CheckCircle2, AlertTriangle
} from 'lucide-react';


interface SystemInfo {
  pytorch_version: string;
  cuda_available: boolean;
  device_name: string;
  gpu_memory: string | null;
  cpu_count: number;
  os_name: string;
  os_release: string;
}

export default function Settings() {
  const [apiUrl, setApiUrl] = useState('');
  const [threshold, setThreshold] = useState(0.60);
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [isLoadingSpecs, setIsLoadingSpecs] = useState(false);
  const [specsError, setSpecsError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  // Load settings on mount
  useEffect(() => {
    setApiUrl(localStorage.getItem('visionai_api_url') || 'http://localhost:8000/api');
    const savedThreshold = localStorage.getItem('visionai_default_threshold');
    setThreshold(savedThreshold ? parseFloat(savedThreshold) : 0.60);
    fetchSystemInfo();
  }, []);

  const fetchSystemInfo = async () => {
    setIsLoadingSpecs(true);
    setSpecsError(null);
    const activeApi = localStorage.getItem('visionai_api_url') || 'http://localhost:8000/api';
    try {
      const res = await axios.get(`${activeApi}/system/info`);
      setSystemInfo(res.data);
    } catch (e: any) {
      console.error(e);
      setSpecsError("Failed to fetch system specifications. Check API URL configuration.");
      setSystemInfo(null);
    } finally {
      setIsLoadingSpecs(false);
    }
  };

  const handleSaveSettings = () => {
    localStorage.setItem('visionai_api_url', apiUrl);
    localStorage.setItem('visionai_default_threshold', threshold.toString());
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
    fetchSystemInfo();
  };

  const architectures = [
    {
      name: 'MobileNetV3',
      type: 'CNN (Lightweight)',
      speed: 'EXCEPTIONAL',
      acc: 'BALANCED',
      desc: 'Optimized for high-speed edge devices and CPU-only hosts. Runs inference in under 15ms.',
      glowing: false
    },
    {
      name: 'EfficientNet-B0',
      type: 'CNN (Efficient)',
      speed: 'HIGH',
      acc: 'GOOD',
      desc: 'Uniform scaling design balancing computation footprint and accuracy. Excellent baseline model.',
      glowing: false
    },
    {
      name: 'ResNet50',
      type: 'CNN (Deep Residual)',
      speed: 'MEDIUM',
      acc: 'EXCELLENT',
      desc: 'Deeper 50-layer architecture mapping complex visual representations. Industry-standard baseline.',
      glowing: false
    },
    {
      name: 'ConvNeXt-Tiny',
      type: 'CNN (Modernized)',
      speed: 'HIGH',
      acc: 'SUPERIOR',
      desc: 'A modernized CNN design that incorporates Vision Transformer strategies. Outstanding robustness.',
      glowing: true
    },
    {
      name: 'ViT-B/16',
      type: 'Vision Transformer',
      speed: 'BALANCED',
      acc: 'SOTA (STATE-OF-THE-ART)',
      desc: 'Splits images into patches and applies self-attention. Highest representational capacity on large data.',
      glowing: true
    }
  ];

  return (
    <div className="space-y-10 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-16">
      {/* Page Header */}
      <header className="relative">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-2 uppercase">
          System Control<span className="text-zinc-500">.</span>
        </h1>
        <p className="text-zinc-400 text-lg tracking-wide max-w-xl font-light font-sans">
          Configure runtime thresholds, network endpoints, and monitor hardware acceleration assets.
        </p>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        
        {/* Connection & Threshold Parameters Card */}
        <div className="xl:col-span-1 space-y-6">
          <div className="glass-panel rounded-3xl p-8 shadow-2xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-48 h-48 bg-white/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />
            
            <h2 className="text-xl font-bold mb-8 flex items-center gap-3 uppercase tracking-widest text-zinc-100">
              <Sliders className="w-6 h-6 text-white" />
              Parameters
            </h2>

            <div className="space-y-6 relative z-10">
              {/* API Endpoints */}
              <div>
                <label className="block text-xs font-bold text-zinc-400 mb-2.5 tracking-[0.1em] uppercase flex items-center gap-1.5">
                  <LinkIcon className="w-3.5 h-3.5" /> API Host URL
                </label>
                <input 
                  type="text" 
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  placeholder="e.g. http://localhost:8000/api"
                  className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-white text-xs font-mono focus:outline-none focus:border-white/30 transition-colors"
                />
              </div>

              {/* Default Threshold */}
              <div>
                <div className="flex justify-between text-xs font-bold tracking-[0.1em] uppercase mb-2.5">
                  <span className="text-zinc-400">Default Confidence Threshold</span>
                  <span className="text-white font-mono">{(threshold * 100).toFixed(0)}%</span>
                </div>
                <input 
                  type="range" 
                  min="0.10" 
                  max="0.95" 
                  step="0.05"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-white"
                />
                <p className="text-[10px] text-zinc-500 mt-2 font-sans font-light leading-relaxed">
                  Predictions scoring lower than this threshold will be flagged as "Unknown / Low Confidence".
                </p>
              </div>

              {/* Save Button */}
              <button 
                onClick={handleSaveSettings}
                className="w-full mt-8 flex items-center justify-center gap-3 bg-white hover:bg-zinc-200 text-black font-extrabold py-3.5 px-4 rounded-xl transition-all shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.2)] tracking-[0.1em] uppercase text-xs"
              >
                {saveSuccess ? (
                  <>
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    Settings Saved
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 fill-current" />
                    Apply Changes
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Hardware Status Specs Card */}
        <div className="xl:col-span-2">
          <div className="glass-panel rounded-3xl p-8 shadow-2xl h-full flex flex-col relative overflow-hidden group">
            <div className="absolute bottom-0 left-0 w-64 h-64 bg-white/5 rounded-full blur-3xl translate-y-1/2 -translate-x-1/4 pointer-events-none" />
            
            <div className="flex justify-between items-center mb-8 relative z-10">
              <h2 className="text-xl font-bold flex items-center gap-3 uppercase tracking-widest text-zinc-100">
                <Cpu className="w-6 h-6 text-white" />
                Hardware Acceleration Telemetry
              </h2>
              
              <button 
                onClick={fetchSystemInfo}
                disabled={isLoadingSpecs}
                className="p-2 bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all rounded-xl text-zinc-400 hover:text-white"
                title="Refresh hardware stats"
              >
                <RefreshCw className={`w-4 h-4 ${isLoadingSpecs ? 'animate-spin' : ''}`} />
              </button>
            </div>

            {systemInfo ? (
              <div className="flex-1 space-y-6 relative z-10">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Compute Device */}
                  <div className="bg-black/40 rounded-2xl p-5 border border-white/5 shadow-inner">
                    <p className="text-[10px] text-zinc-500 font-bold tracking-[0.2em] mb-2 uppercase">Active Compute Device</p>
                    <p className="text-lg font-extrabold text-white truncate font-mono">{systemInfo.device_name}</p>
                  </div>

                  {/* GPU Memory */}
                  <div className="bg-black/40 rounded-2xl p-5 border border-white/5 shadow-inner">
                    <p className="text-[10px] text-zinc-500 font-bold tracking-[0.2em] mb-2 uppercase">Available GPU Framebuffer</p>
                    <p className="text-lg font-extrabold text-white font-mono">{systemInfo.gpu_memory || 'N/A (CPU Mode)'}</p>
                  </div>

                  {/* CUDA Status */}
                  <div className="bg-black/40 rounded-2xl p-5 border border-white/5 shadow-inner">
                    <p className="text-[10px] text-zinc-500 font-bold tracking-[0.2em] mb-2 uppercase">CUDA Acceleration Core</p>
                    <div className="flex items-center gap-2 mt-1">
                      <div className={`w-2.5 h-2.5 rounded-full ${systemInfo.cuda_available ? 'bg-emerald-400 shadow-[0_0_10px_#10b981]' : 'bg-zinc-600'}`} />
                      <span className="text-sm font-extrabold uppercase font-mono tracking-wider text-white">
                        {systemInfo.cuda_available ? 'ONLINE / ENGAGED' : 'OFFLINE (FALLBACK CPU)'}
                      </span>
                    </div>
                  </div>

                  {/* PyTorch Version */}
                  <div className="bg-black/40 rounded-2xl p-5 border border-white/5 shadow-inner">
                    <p className="text-[10px] text-zinc-500 font-bold tracking-[0.2em] mb-2 uppercase">PyTorch Engine Version</p>
                    <p className="text-sm font-extrabold text-zinc-300 font-mono mt-1">{systemInfo.pytorch_version}</p>
                  </div>
                </div>

                <div className="pt-6 border-t border-white/5 flex flex-col md:flex-row justify-between text-xs text-zinc-500 font-mono gap-4">
                  <div>OS COMPATIBILITY: {systemInfo.os_name} {systemInfo.os_release}</div>
                  <div>LOGICAL COMPUTE THREADS: {systemInfo.cpu_count} THREADS</div>
                </div>
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-500 relative z-10 py-10">
                {specsError ? (
                  <div className="flex flex-col items-center max-w-sm">
                    <AlertTriangle className="w-12 h-12 text-amber-500 mb-4" />
                    <p className="font-bold text-sm text-zinc-400 text-center uppercase tracking-wide mb-2">Telemetry Pipeline Offline</p>
                    <p className="text-xs text-zinc-600 text-center leading-relaxed font-sans">{specsError}</p>
                  </div>
                ) : (
                  <>
                    <Cpu className="w-16 h-16 mb-4 opacity-10 animate-pulse" />
                    <p className="font-light tracking-[0.2em] uppercase text-xs">Querying System Core Specs...</p>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Network Architectures Catalog */}
      <section className="space-y-6">
        <h2 className="text-xl font-bold flex items-center gap-3 uppercase tracking-widest text-zinc-100">
          <Layers className="w-6 h-6 text-white" />
          Supported Neural Architectures
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {architectures.map((arch) => (
            <div 
              key={arch.name} 
              className={`glass-panel p-6 rounded-3xl border transition-all duration-300 flex flex-col justify-between ${
                arch.glowing 
                  ? 'border-white/20 shadow-[0_0_20px_rgba(255,255,255,0.05)] bg-white/[0.01]' 
                  : 'border-white/5 bg-black/40'
              }`}
            >
              <div>
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-lg font-extrabold text-white tracking-wide">{arch.name}</h3>
                    <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider mt-0.5">{arch.type}</p>
                  </div>
                  {arch.glowing && (
                    <span className="text-[8px] bg-white/10 text-white border border-white/20 px-2 py-0.5 rounded-full font-extrabold tracking-widest uppercase">
                      Premium
                    </span>
                  )}
                </div>
                <p className="text-xs text-zinc-400 font-sans font-light leading-relaxed mb-6">{arch.desc}</p>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-4 border-t border-white/5 text-[10px] font-mono">
                <div>
                  <span className="text-zinc-600 block uppercase font-sans">INFERENCE LATENCY</span>
                  <span className="text-zinc-300 font-bold">{arch.speed}</span>
                </div>
                <div>
                  <span className="text-zinc-600 block uppercase font-sans">ACCURACY RATING</span>
                  <span className="text-zinc-300 font-bold">{arch.acc}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
