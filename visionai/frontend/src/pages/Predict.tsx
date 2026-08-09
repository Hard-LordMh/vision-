import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import axios from 'axios';
import { 
  Play, Image as ImageIcon, CheckCircle, ShieldAlert, Clock, Loader2, Sparkles, 
  FileArchive, Search, Filter, Maximize2, X, Zap, BarChart3, Grid, Layers, Sliders
} from 'lucide-react';

import { getApiUrl, getDefaultThreshold } from '../config';

interface SinglePredictionResult {
  predicted_class: string;
  confidence: number;
  top_3: { class: string; confidence: number }[];
  inference_time_ms: number;
  model_version: string;
}

interface ZipPredictionItem {
  filename: string;
  predicted_class: string;
  confidence: number;
  top_3: { class: string; confidence: number }[];
  inference_time_ms: number;
  image_data_url?: string;
}

interface ZipPredictionResult {
  filename: string;
  total_images: number;
  total_inference_time_ms: number;
  class_summary: Record<string, number>;
  model_version: string;
  predictions: ZipPredictionItem[];
}

export default function Predict() {
  const location = useLocation();
  
  // Config State
  const [apiUrl, setApiUrl] = useState(getApiUrl());
  const [defaultThreshold, setDefaultThreshold] = useState(getDefaultThreshold());

  // Mode selection: 'single' | 'zip'
  const [mode, setMode] = useState<'single' | 'zip'>('single');
  const [activeModel, setActiveModel] = useState<any>(null);
  const [samples, setSamples] = useState<any[]>([]);
  const [sampleZipUrl, setSampleZipUrl] = useState<string | null>(null);

  // --- Single Mode State ---
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [singleResult, setSingleResult] = useState<SinglePredictionResult | null>(null);
  const [loadingSample, setLoadingSample] = useState<string | null>(null);

  // --- ZIP Mode State ---
  const [selectedZip, setSelectedZip] = useState<File | null>(null);
  const [zipFileName, setZipFileName] = useState<string | null>(null);
  const [isPredictingZip, setIsPredictingZip] = useState(false);
  const [loadingZipSample, setLoadingZipSample] = useState(false);
  const [zipResult, setZipResult] = useState<ZipPredictionResult | null>(null);
  
  // Scanner animation step message
  const [scanStep, setScanStep] = useState(0);

  // Filter & Search & Modal in ZIP result view
  const [searchQuery, setSearchQuery] = useState('');
  const [filterClass, setFilterClass] = useState<string>('ALL');
  const [sortBy, setSortBy] = useState<'confidence_desc' | 'confidence_asc' | 'name'>('confidence_desc');
  const [inspectedItem, setInspectedItem] = useState<ZipPredictionItem | null>(null);

  // --- Tabular ANN State ---
  const [projectMode, setProjectMode] = useState<'vision' | 'placement'>(() => {
    return (localStorage.getItem('visionai_project_mode') as 'vision' | 'placement') || 'vision';
  });
  const [cgpa, setCgpa] = useState(8.0);
  const [aptitudeScore, setAptitudeScore] = useState(75);
  const [communicationSkills, setCommunicationSkills] = useState(4);
  const [codingSkills, setCodingSkills] = useState(4);
  const [internship, setInternship] = useState<'Yes' | 'No'>('No');
  const [projectsCompleted, setProjectsCompleted] = useState(2);
  const [tabularPredicting, setTabularPredicting] = useState(false);
  const [tabularResult, setTabularResult] = useState<any>(null);

  const fetchActiveModel = (modeVal: 'vision' | 'placement', activeApi: string) => {
    axios.get(`${activeApi}/models`).then(res => {
      const isPlacement = modeVal === 'placement';
      const allModels: any[] = res.data;
      
      // Filter by type (ANN = placement, other = vision)
      const typedModels = allModels.filter((m: any) => {
        const isModelPlacement = m.architecture.startsWith('ANN');
        return isPlacement ? isModelPlacement : !isModelPlacement;
      });
      
      // Prefer active models, then any model of the right type (latest first by id)
      const activeOnes = typedModels.filter((m: any) => m.is_active);
      const best = activeOnes.length > 0
        ? activeOnes.reduce((prev: any, cur: any) => (cur.id > prev.id ? cur : prev))
        : typedModels.length > 0
          ? typedModels.reduce((prev: any, cur: any) => (cur.id > prev.id ? cur : prev))
          : null;
      
      setActiveModel(best);
    }).catch(err => console.log('Models fetch error:', err));
  };

  const resolveStaticUrl = (path: string) => {
    if (!path) return '';
    if (path.startsWith('http')) return path;
    const baseApiUrl = apiUrl.endsWith('/api') ? apiUrl.slice(0, -4) : apiUrl.replace('/api/', '/').replace('/api', '');
    return `${baseApiUrl}${path}`;
  };

  useEffect(() => {
    const activeApi = getApiUrl();
    setApiUrl(activeApi);
    setDefaultThreshold(getDefaultThreshold());
    
    fetchActiveModel(projectMode, activeApi);

    axios.get(`${activeApi}/samples/info`).then(res => {
      if (res.data) {
        if (res.data.samples) setSamples(res.data.samples);
        if (res.data.dataset_zip_url) setSampleZipUrl(res.data.dataset_zip_url);
      }
    }).catch(err => console.log('Sample info fetch error:', err));

    const handleModeChange = () => {
      const newMode = (localStorage.getItem('visionai_project_mode') as 'vision' | 'placement') || 'vision';
      setProjectMode(newMode);
      fetchActiveModel(newMode, activeApi);
    };
    window.addEventListener('projectModeChanged', handleModeChange);
    
    return () => window.removeEventListener('projectModeChanged', handleModeChange);
  }, [projectMode]);

  // Handle direct navigation state from Datasets page if passed
  useEffect(() => {
    if (location.state?.mode === 'zip') {
      setMode('zip');
      if (location.state?.autoLoadPreloaded) {
        handleLoadPreloadedZip();
      }
    }
  }, [location.state]);

  // --- Scanner status loop when predicting ZIP ---
  useEffect(() => {
    if (!isPredictingZip) {
      setScanStep(0);
      return;
    }
    const interval = setInterval(() => {
      setScanStep(prev => (prev + 1) % scanMessages.length);
    }, 800);
    return () => clearInterval(interval);
  }, [isPredictingZip]);

  const scanMessages = [
    "UNPACKING ZIP ARCHIVE BLOCKS...",
    "EXTRACTING IMAGE MATRIX TELEMETRY...",
    "INITIALIZING NEURAL TENSOR CONSTRUCTS...",
    "EXECUTING FORWARD PASS CLASSIFICATION...",
    "GENERATING BASE64 THUMBNAIL SPECS...",
    "SYNTHESIZING MATRIX SUMMARY REPORT..."
  ];

  // Single image logic
  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setSingleResult(null);
    }
  };



  // Auto-select sample AND immediately run prediction
  const handleSelectAndPredict = async (sample: any) => {
    try {
      setLoadingSample(sample.id);
      const sampleUrl = resolveStaticUrl(sample.url);
      const response = await fetch(sampleUrl);
      const blob = await response.blob();
      const file = new File([blob], sample.filename, { type: 'image/jpeg' });
      setSelectedImage(file);
      setPreviewUrl(sampleUrl);
      setSingleResult(null);
      setLoadingSample(null);
      // Immediately trigger prediction
      setIsPredicting(true);
      const formData = new FormData();
      formData.append('file', file);
      formData.append('threshold', defaultThreshold.toString());
      try {
        const res = await axios.post(`${apiUrl}/predict`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setSingleResult(res.data);
      } catch (e: any) {
        console.error(e);
        alert(e.response?.data?.detail || "Prediction failed");
      } finally {
        setIsPredicting(false);
      }
    } catch (e) {
      console.error("Failed to load sample image", e);
      alert("Failed to load sample image from server.");
      setLoadingSample(null);
    }
  };

  const handlePredictSingle = async () => {
    if (!selectedImage) return;
    
    setIsPredicting(true);
    const formData = new FormData();
    formData.append('file', selectedImage);
    formData.append('threshold', defaultThreshold.toString());

    try {
      const res = await axios.post(`${apiUrl}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setSingleResult(res.data);
    } catch (e: any) {
      console.error(e);
      alert(e.response?.data?.detail || "Prediction failed");
    } finally {
      setIsPredicting(false);
    }
  };

  // ZIP logic
  const handleZipSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (!file.name.endsWith('.zip')) {
        alert("Only .zip files are supported.");
        return;
      }
      setSelectedZip(file);
      setZipFileName(file.name);
      setZipResult(null);
    }
  };

  const handleLoadPreloadedZip = async () => {
    try {
      setLoadingZipSample(true);
      const zipUrl = resolveStaticUrl(sampleZipUrl || '/static/samples/sample_animals_dataset.zip');
      const response = await fetch(zipUrl);
      const blob = await response.blob();
      const file = new File([blob], 'sample_animals_dataset.zip', { type: 'application/zip' });
      setSelectedZip(file);
      setZipFileName(file.name);
      setZipResult(null);
    } catch (e) {
      console.error("Failed to fetch preloaded zip", e);
      alert("Failed to load preloaded sample ZIP.");
    } finally {
      setLoadingZipSample(false);
    }
  };

  const handlePredictZip = async (fileToPredict?: File) => {
    const file = fileToPredict || selectedZip;
    if (!file) return;

    setIsPredictingZip(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('threshold', defaultThreshold.toString());

    try {
      const res = await axios.post(`${apiUrl}/predict/zip`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setZipResult(res.data);
      setFilterClass('ALL');
    } catch (e: any) {
      console.error(e);
      alert(e.response?.data?.detail || "ZIP Batch Prediction failed.");
    } finally {
      setIsPredictingZip(false);
    }
  };

  // Filtered & Sorted ZIP Predictions
  const getFilteredZipPredictions = () => {
    if (!zipResult) return [];
    let items = [...zipResult.predictions];

    if (filterClass !== 'ALL') {
      items = items.filter(item => item.predicted_class === filterClass);
    }

    if (searchQuery.trim() !== '') {
      const q = searchQuery.toLowerCase();
      items = items.filter(item => 
        item.filename.toLowerCase().includes(q) || 
        item.predicted_class.toLowerCase().includes(q)
      );
    }

    if (sortBy === 'confidence_desc') {
      items.sort((a, b) => b.confidence - a.confidence);
    } else if (sortBy === 'confidence_asc') {
      items.sort((a, b) => a.confidence - b.confidence);
    } else if (sortBy === 'name') {
      items.sort((a, b) => a.filename.localeCompare(b.filename));
    }

    return items;
  };

  const handlePredictTabular = async () => {
    setTabularPredicting(true);
    try {
      const res = await axios.post(`${apiUrl}/predict/tabular`, {
        cgpa,
        aptitude_score: aptitudeScore,
        communication_skills: communicationSkills,
        coding_skills: codingSkills,
        internship,
        projects_completed: projectsCompleted
      });
      setTabularResult(res.data);
    } catch (e: any) {
      console.error(e);
      alert(e.response?.data?.detail || "Tabular prediction failed");
    } finally {
      setTabularPredicting(false);
    }
  };

  return (
    <div className="space-y-10 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-16">
      {/* Top Header */}
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-6 relative z-10">
        <div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-2 uppercase">
            {projectMode === 'vision' ? 'Inference Engine' : 'Placement Inference'}<span className="text-zinc-500">.</span>
          </h1>
          <p className="text-zinc-400 text-lg tracking-wide max-w-xl font-light">
            {projectMode === 'vision'
              ? 'Real-time neural matrix classification for single imagery & batch archive packages.'
              : 'Real-time student placement probability prediction using Artificial Neural Network.'}
          </p>
        </div>
        
        {/* Active Model Indicator */}
        {activeModel ? (
          <div className="flex items-center gap-3 bg-white/10 border border-white/20 px-5 py-2.5 rounded-2xl text-white font-bold text-xs tracking-widest uppercase shadow-[0_0_15px_rgba(255,255,255,0.1)] backdrop-blur-md self-start md:self-auto">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
            <span className="text-zinc-400 mr-1">ACTIVE:</span> {activeModel.name} <span className="opacity-50">v{activeModel.version}</span>
          </div>
        ) : (
          <div className="flex items-center gap-3 bg-zinc-800 border border-zinc-600 px-5 py-2.5 rounded-2xl text-zinc-300 font-bold text-xs tracking-widest uppercase shadow-inner backdrop-blur-md">
            <ShieldAlert className="w-4 h-4 text-amber-400" />
            System Offline - No Active Model
          </div>
        )}
      </header>

      {/* Mode Selector Navigation Pills */}
      {projectMode === 'vision' && (
        <div className="flex items-center justify-between bg-black/60 border border-white/10 p-1.5 rounded-2xl max-w-xl">
          <button
            id="mode-single-btn"
            onClick={() => setMode('single')}
            className={`flex-1 flex items-center justify-center gap-3 py-3 px-6 rounded-xl font-extrabold text-xs tracking-[0.15em] uppercase transition-all duration-300 ${
              mode === 'single'
                ? 'bg-white text-black shadow-[0_0_20px_rgba(255,255,255,0.2)]'
                : 'text-zinc-400 hover:text-white hover:bg-white/5'
            }`}
          >
            <ImageIcon className="w-4 h-4" />
            Single Telemetry Target
          </button>

          <button
            id="mode-zip-btn"
            onClick={() => setMode('zip')}
            className={`flex-1 flex items-center justify-center gap-3 py-3 px-6 rounded-xl font-extrabold text-xs tracking-[0.15em] uppercase transition-all duration-300 relative ${
              mode === 'zip'
                ? 'bg-white text-black shadow-[0_0_20px_rgba(255,255,255,0.2)]'
                : 'text-zinc-400 hover:text-white hover:bg-white/5'
            }`}
          >
            <FileArchive className="w-4 h-4 text-amber-400" />
            ZIP Archive Matrix
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
            </span>
          </button>
        </div>
      )}


      {/* ========================================================================= */}
      {/* MODE 1: SINGLE IMAGE PREDICTION MODE */}
      {/* ========================================================================= */}
      {projectMode === 'vision' && mode === 'single' && (
        <div className="space-y-10 animate-in fade-in duration-500">
          {/* Preloaded Quick Samples Bar */}
          {samples.length > 0 && (
            <div className="glass-panel p-6 rounded-3xl border border-white/10 relative overflow-hidden">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2 text-xs font-extrabold tracking-[0.2em] uppercase text-zinc-300">
                  <Sparkles className="w-4 h-4 text-amber-400 animate-pulse" />
                  Preloaded Sample Telemetry — Click to Load &amp; Analyse
                </div>
                <span className="text-[10px] bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 px-3 py-1 rounded-full font-bold uppercase tracking-widest flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping inline-block" />
                  Click = Instant Analysis
                </span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {samples.map((sample) => (
                  <button
                    key={sample.id}
                    onClick={() => handleSelectAndPredict(sample)}
                    disabled={loadingSample === sample.id || isPredicting}
                    className="flex items-center gap-4 p-3 bg-black/40 hover:bg-white/10 border border-white/10 hover:border-amber-400/40 rounded-2xl transition-all duration-300 text-left group relative overflow-hidden"
                  >
                    {/* Shimmer on hover */}
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-amber-400/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700 pointer-events-none" />
                    <div className="w-14 h-14 rounded-xl overflow-hidden bg-zinc-900 border border-white/10 shrink-0 relative">
                      <img src={resolveStaticUrl(sample.url)} alt={sample.label} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" />
                      {loadingSample === sample.id && (
                        <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                          <Loader2 className="w-5 h-5 text-amber-400 animate-spin" />
                        </div>
                      )}
                    </div>
                    <div className="flex-1">
                      <h4 className="text-sm font-bold text-white group-hover:text-amber-300 transition-colors">{sample.label}</h4>
                      <p className="text-[10px] text-zinc-400 uppercase font-mono tracking-wider mt-0.5">Target: {sample.class_name}</p>
                      <p className="text-[10px] text-amber-500/70 uppercase font-mono tracking-wider mt-1 group-hover:text-amber-400 transition-colors flex items-center gap-1">
                        <Play className="w-2.5 h-2.5 fill-current" /> Click to analyse
                      </p>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
            {/* Upload Area */}
            <div className="glass-panel rounded-3xl p-8 shadow-2xl flex flex-col h-[500px] relative overflow-hidden group transition-all duration-300">
              <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />
              <h2 className="text-xl font-bold mb-6 flex items-center gap-3 uppercase tracking-widest text-zinc-100 relative z-10">
                <ImageIcon className="w-6 h-6 text-white" />
                Input Telemetry
              </h2>
              
              <div className="flex-1 relative border-2 border-dashed border-white/10 hover:border-white/30 rounded-2xl transition-all duration-300 bg-white/[0.02] group-hover:bg-white/[0.04] overflow-hidden flex items-center justify-center z-10 shadow-inner">
                {previewUrl ? (
                  <img src={previewUrl} alt="Preview" className="w-full h-full object-contain p-2" />
                ) : (
                  <div className="flex flex-col items-center text-zinc-500 group-hover:text-white transition-colors pointer-events-none">
                    <ImageIcon className="w-16 h-16 mb-4 drop-shadow-[0_0_15px_rgba(255,255,255,0)] group-hover:drop-shadow-[0_0_15px_rgba(255,255,255,0.2)] transition-all" />
                    <p className="font-bold tracking-widest uppercase text-sm text-zinc-300 group-hover:text-white">Target Image For Analysis</p>
                    <p className="text-xs mt-2 font-light tracking-wide text-zinc-600 group-hover:text-zinc-400">JPG, PNG, WEBP Supported or Select Preloaded Sample Above</p>
                  </div>
                )}
                <input 
                  type="file" 
                  accept="image/*"
                  onChange={handleImageSelect}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
              </div>

              <button 
                id="execute-single-predict-btn"
                onClick={handlePredictSingle}
                disabled={!selectedImage || isPredicting}
                className="w-full mt-6 flex items-center justify-center gap-3 bg-white hover:bg-zinc-200 disabled:bg-white/5 disabled:text-zinc-500 text-black font-extrabold py-4 px-4 rounded-xl transition-all shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_30px_rgba(255,255,255,0.2)] disabled:shadow-none tracking-[0.1em] uppercase text-sm relative z-10"
              >
                {isPredicting ? (
                  <><Loader2 className="w-5 h-5 animate-spin" /> Processing Matrix...</>
                ) : (
                  <><Play className="w-5 h-5 fill-current" /> Execute Inference</>
                )}
              </button>
            </div>

            {/* Prediction Results */}
            <div className="glass-panel rounded-3xl p-8 shadow-2xl relative overflow-hidden group transition-all duration-300">
              <div className="absolute bottom-0 left-0 w-64 h-64 bg-white/5 rounded-full blur-3xl translate-y-1/2 -translate-x-1/4 pointer-events-none" />
              <h2 className="text-xl font-bold mb-8 flex items-center gap-3 uppercase tracking-widest text-zinc-100 relative z-10">
                <CheckCircle className="w-6 h-6 text-white" />
                Analysis Results
              </h2>

              {singleResult ? (
                <div className="space-y-8 animate-in slide-in-from-right-8 duration-500 relative z-10">
                  <div className="text-center p-8 bg-black/40 border border-white/5 rounded-3xl shadow-inner backdrop-blur-sm">
                    <p className="text-xs text-zinc-400 font-bold tracking-[0.2em] uppercase mb-4">Primary Classification</p>
                    <h3 className={`text-5xl font-extrabold tracking-tight ${singleResult.predicted_class === "Unknown / Low Confidence" ? 'text-zinc-500' : 'text-white text-shadow-[0_0_15px_rgba(255,255,255,0.3)]'}`}>
                      {singleResult.predicted_class}
                    </h3>
                    
                    <div className="mt-8">
                      <div className="flex justify-between text-xs font-bold tracking-widest uppercase mb-3">
                        <span className="text-zinc-500">Confidence Rating</span>
                        <span className="text-white">{(singleResult.confidence * 100).toFixed(2)}%</span>
                      </div>
                      <div className="h-4 w-full bg-black/60 rounded-full overflow-hidden border border-white/10 shadow-inner">
                        <div 
                          className={`h-full transition-all duration-1000 ${singleResult.predicted_class === "Unknown / Low Confidence" ? 'bg-zinc-600' : 'bg-white shadow-[0_0_20px_rgba(255,255,255,0.5)]'}`}
                          style={{ width: `${singleResult.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-xs font-bold text-zinc-500 tracking-[0.1em] uppercase mb-4">Secondary Probabilities</h4>
                    <div className="space-y-3">
                      {singleResult.top_3.map((pred: any, i: number) => (
                        <div key={i} className="flex justify-between items-center p-4 bg-white/5 rounded-xl border border-white/5 hover:border-white/30 transition-colors">
                          <span className="font-bold tracking-wide text-zinc-300">{pred.class}</span>
                          <span className="text-white font-mono font-bold">{(pred.confidence * 100).toFixed(2)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center gap-6 pt-8 border-t border-white/10">
                    <div className="flex items-center gap-3 text-xs font-bold tracking-widest uppercase text-zinc-500">
                      <Clock className="w-5 h-5 text-zinc-400" />
                      Latency: <span className="font-mono text-white ml-1">{singleResult.inference_time_ms.toFixed(1)} ms</span>
                    </div>
                    <div className="text-zinc-800 text-xl font-light">|</div>
                    <div className="text-xs font-bold tracking-widest uppercase text-zinc-500">
                      Version: <span className="text-white ml-2">{singleResult.model_version}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-zinc-600 min-h-[350px] relative z-10">
                  <TargetPlaceholder />
                  <p className="mt-8 font-light tracking-[0.2em] uppercase text-sm">System standing by for telemetry...</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}


      {/* ========================================================================= */}
      {/* MODE 2: ZIP ARCHIVE BATCH PREDICTION MODE */}
      {/* ========================================================================= */}
      {projectMode === 'vision' && mode === 'zip' && (
        <div className="space-y-10 animate-in fade-in duration-500">
          
          {/* Quick Instant Test Zip Action Bar */}
          <div className="glass-panel p-6 rounded-3xl border border-amber-500/20 bg-amber-500/5 relative overflow-hidden flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-4 relative z-10">
              <div className="p-3.5 bg-amber-500/20 border border-amber-500/30 rounded-2xl text-amber-400 shrink-0">
                <Sparkles className="w-6 h-6 animate-pulse" />
              </div>
              <div>
                <h3 className="text-lg font-extrabold text-white uppercase tracking-wider flex items-center gap-2">
                  Preloaded Animal ZIP Matrix Test
                  <span className="text-[10px] bg-amber-400/20 text-amber-300 border border-amber-400/30 px-2 py-0.5 rounded-full font-mono">Instant Batch Test</span>
                </h3>
                <p className="text-xs text-zinc-400 font-light mt-0.5">
                  Contains multiple sample images (Cat, Dog, Bird). Predict all contents simultaneously with 1 click.
                </p>
              </div>
            </div>

            <button
              id="load-sample-zip-btn"
              onClick={async () => {
                setLoadingZipSample(true);
                try {
                  const zipUrl = resolveStaticUrl(sampleZipUrl || '/static/samples/sample_animals_dataset.zip');
                  const response = await fetch(zipUrl);
                  const blob = await response.blob();
                  const file = new File([blob], 'sample_animals_dataset.zip', { type: 'application/zip' });
                  setSelectedZip(file);
                  setZipFileName(file.name);
                  setZipResult(null);
                  handlePredictZip(file);
                } catch(e) {
                  alert('Failed to load preloaded sample ZIP.');
                } finally {
                  setLoadingZipSample(false);
                }
              }}
              disabled={loadingZipSample || isPredictingZip}
              className="w-full md:w-auto flex items-center justify-center gap-3 bg-gradient-to-r from-amber-400 to-amber-500 hover:from-amber-300 hover:to-amber-400 text-black font-extrabold text-xs px-6 py-3.5 rounded-xl uppercase tracking-widest transition-all shadow-[0_0_20px_rgba(245,158,11,0.3)] disabled:opacity-50 shrink-0"
            >
              {loadingZipSample || isPredictingZip ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Zap className="w-4 h-4 fill-current" />
              )}
              {loadingZipSample ? "Loading Archive..." : isPredictingZip ? "Scanning..." : "Run Preloaded ZIP Prediction"}
            </button>
          </div>

          {/* Upload Drop Zone for Custom ZIP */}
          <div className="glass-panel rounded-3xl p-8 shadow-2xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-96 h-96 bg-amber-500/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />
            
            <h2 className="text-xl font-bold mb-6 flex items-center gap-3 uppercase tracking-widest text-zinc-100 relative z-10">
              <FileArchive className="w-6 h-6 text-amber-400" />
              Upload ZIP Telemetry Archive
            </h2>

            <div className="relative border-2 border-dashed border-amber-500/20 hover:border-amber-400/40 rounded-2xl p-10 text-center transition-all duration-300 bg-black/40 overflow-hidden group shadow-inner">
              <input 
                id="zip-file-input"
                type="file" 
                accept=".zip"
                onChange={handleZipSelect}
                disabled={isPredictingZip}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed z-20"
              />

              <div className="flex flex-col items-center justify-center relative z-10">
                <FileArchive className="w-16 h-16 mb-4 text-amber-400/70 group-hover:text-amber-400 group-hover:scale-110 transition-all duration-300 drop-shadow-[0_0_15px_rgba(245,158,11,0.2)]" />
                
                {zipFileName ? (
                  <div className="space-y-2">
                    <p className="font-extrabold tracking-widest text-white text-lg font-mono uppercase bg-amber-500/20 px-4 py-1.5 rounded-xl border border-amber-500/30 inline-block">
                      {zipFileName}
                    </p>
                    <p className="text-xs text-zinc-400 font-light">Archive ready for neural scan execution.</p>
                  </div>
                ) : (
                  <div>
                    <p className="font-extrabold tracking-widest uppercase text-base text-zinc-200 group-hover:text-white transition-colors">
                      Drag & Drop (.ZIP) Archive File
                    </p>
                    <p className="text-xs mt-2 font-light tracking-wide text-zinc-500 group-hover:text-zinc-400">
                      Standard ZIP container containing target imagery (.JPG, .PNG, .WEBP)
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center justify-between gap-4 mt-6 relative z-10">
              <button 
                id="execute-zip-predict-btn"
                onClick={() => handlePredictZip()}
                disabled={!selectedZip || isPredictingZip}
                className="flex-1 flex items-center justify-center gap-3 bg-white hover:bg-zinc-200 disabled:bg-white/5 disabled:text-zinc-600 text-black font-extrabold py-4 px-6 rounded-xl transition-all shadow-[0_0_25px_rgba(255,255,255,0.15)] disabled:shadow-none tracking-[0.15em] uppercase text-sm"
              >
                {isPredictingZip ? (
                  <><Loader2 className="w-5 h-5 animate-spin text-black" /> Scanning Neural Matrix...</>
                ) : (
                  <><Play className="w-5 h-5 fill-current" /> Execute ZIP Archive Prediction</>
                )}
              </button>

              {zipFileName && (
                <button
                  onClick={() => {
                    setSelectedZip(null);
                    setZipFileName(null);
                    setZipResult(null);
                  }}
                  disabled={isPredictingZip}
                  className="bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white px-4 py-4 rounded-xl text-xs font-bold uppercase tracking-widest border border-white/10 transition-colors"
                >
                  Clear Archive
                </button>
              )}
            </div>
          </div>


          {/* Cybernetic Scanner Loading State */}
          {isPredictingZip && (
            <div className="glass-panel p-12 rounded-3xl border border-amber-500/30 bg-black/80 flex flex-col items-center justify-center text-center relative overflow-hidden animate-in fade-in duration-300">
              {/* Laser Scanning Bar */}
              <div className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-amber-400 to-transparent top-0 animate-[progress_2s_ease-in-out_infinite] shadow-[0_0_15px_#f59e0b]" />
              
              {/* Radar Reticle Animation */}
              <div className="relative w-32 h-32 mb-8 flex items-center justify-center">
                <div className="absolute inset-0 rounded-full border-2 border-amber-500/20 animate-ping" />
                <div className="absolute inset-2 rounded-full border border-dashed border-amber-400/40 animate-[spin_10s_linear_infinite]" />
                <div className="absolute inset-6 rounded-full border border-amber-300/30 animate-[spin_6s_linear_infinite_reverse]" />
                <FileArchive className="w-10 h-10 text-amber-400 animate-pulse" />
              </div>

              <h3 className="text-2xl font-extrabold text-white uppercase tracking-[0.2em] mb-2">
                Neural Scanning In Progress
              </h3>
              
              <div className="font-mono text-xs text-amber-400 bg-amber-500/10 border border-amber-500/30 px-4 py-2 rounded-xl tracking-widest uppercase mb-4 animate-pulse">
                {scanMessages[scanStep]}
              </div>

              <p className="text-xs text-zinc-500 tracking-wider uppercase font-light">
                Processing ZIP batch telemetry through active model architecture
              </p>
            </div>
          )}


          {/* ========================================================================= */}
          {/* ADANIMATION VISUAL MATRIX GALLERY RESULTS */}
          {/* ========================================================================= */}
          {zipResult && !isPredictingZip && (
            <div className="space-y-8 animate-in fade-in slide-in-from-bottom-6 duration-700">
              
              {/* Top HUD Overview Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                
                <div className="glass-panel p-6 rounded-3xl border border-white/10 flex items-center gap-5">
                  <div className="p-4 bg-white/10 rounded-2xl text-white border border-white/20">
                    <Grid className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="text-[10px] text-zinc-400 font-bold uppercase tracking-[0.2em]">Scanned Telemetry</p>
                    <h4 className="text-3xl font-extrabold text-white mt-1 font-mono">{zipResult.total_images} <span className="text-xs font-light text-zinc-500 font-sans">IMGS</span></h4>
                  </div>
                </div>

                <div className="glass-panel p-6 rounded-3xl border border-white/10 flex items-center gap-5">
                  <div className="p-4 bg-amber-500/10 rounded-2xl text-amber-400 border border-amber-500/20">
                    <Clock className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="text-[10px] text-zinc-400 font-bold uppercase tracking-[0.2em]">Batch Latency</p>
                    <h4 className="text-3xl font-extrabold text-white mt-1 font-mono">{zipResult.total_inference_time_ms.toFixed(1)} <span className="text-xs font-light text-zinc-500 font-sans">MS</span></h4>
                  </div>
                </div>

                <div className="glass-panel p-6 rounded-3xl border border-white/10 flex items-center gap-5">
                  <div className="p-4 bg-emerald-500/10 rounded-2xl text-emerald-400 border border-emerald-500/20">
                    <Zap className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="text-[10px] text-zinc-400 font-bold uppercase tracking-[0.2em]">Avg Speed / Img</p>
                    <h4 className="text-3xl font-extrabold text-white mt-1 font-mono">
                      {(zipResult.total_inference_time_ms / (zipResult.total_images || 1)).toFixed(1)} <span className="text-xs font-light text-zinc-500 font-sans">MS</span>
                    </h4>
                  </div>
                </div>

                <div className="glass-panel p-6 rounded-3xl border border-white/10 flex items-center gap-5">
                  <div className="p-4 bg-indigo-500/10 rounded-2xl text-indigo-400 border border-indigo-500/20">
                    <Layers className="w-6 h-6" />
                  </div>
                  <div>
                    <p className="text-[10px] text-zinc-400 font-bold uppercase tracking-[0.2em]">Class Matrix</p>
                    <h4 className="text-3xl font-extrabold text-white mt-1 font-mono">{Object.keys(zipResult.class_summary).length} <span className="text-xs font-light text-zinc-500 font-sans">TYPES</span></h4>
                  </div>
                </div>

              </div>

              {/* Class Summary Filter Breakdown Bar */}
              <div className="glass-panel p-6 rounded-3xl border border-white/10 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-extrabold text-zinc-300 uppercase tracking-[0.2em] flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-amber-400" />
                    Classification Distribution Matrix
                  </h3>
                  <span className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest">
                    Click filter tag to isolate
                  </span>
                </div>

                <div className="flex flex-wrap gap-3">
                  <button
                    onClick={() => setFilterClass('ALL')}
                    className={`px-4 py-2 rounded-xl text-xs font-extrabold tracking-widest uppercase transition-all border ${
                      filterClass === 'ALL'
                        ? 'bg-white text-black border-white shadow-[0_0_15px_rgba(255,255,255,0.3)]'
                        : 'bg-black/40 text-zinc-400 border-white/10 hover:border-white/30 hover:text-white'
                    }`}
                  >
                    ALL TARGETS ({zipResult.total_images})
                  </button>

                  {Object.entries(zipResult.class_summary).map(([className, count]) => {
                    const pct = ((count / zipResult.total_images) * 100).toFixed(1);
                    return (
                      <button
                        key={className}
                        onClick={() => setFilterClass(className)}
                        className={`flex items-center gap-3 px-4 py-2 rounded-xl text-xs font-bold tracking-widest uppercase transition-all border ${
                          filterClass === className
                            ? 'bg-amber-400 text-black border-amber-400 shadow-[0_0_15px_rgba(245,158,11,0.4)]'
                            : 'bg-white/5 text-zinc-300 border-white/10 hover:border-white/30 hover:bg-white/10'
                        }`}
                      >
                        <span>{className}</span>
                        <span className={`px-2 py-0.5 rounded-md text-[10px] font-mono ${filterClass === className ? 'bg-black/30 text-black' : 'bg-white/10 text-white'}`}>
                          {count} ({pct}%)
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Filter & Search Bar */}
              <div className="flex flex-col md:flex-row items-center justify-between gap-4">
                {/* Search Bar */}
                <div className="relative w-full md:w-96">
                  <Search className="w-4 h-4 text-zinc-500 absolute left-4 top-1/2 -translate-y-1/2" />
                  <input
                    id="zip-search-input"
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search file name or class..."
                    className="w-full bg-black/60 border border-white/10 rounded-2xl pl-11 pr-4 py-3 text-xs text-white placeholder:text-zinc-600 focus:outline-none focus:border-white/30 transition-colors"
                  />
                  {searchQuery && (
                    <button onClick={() => setSearchQuery('')} className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-white text-xs">
                      Clear
                    </button>
                  )}
                </div>

                {/* Sort Order dropdown */}
                <div className="flex items-center gap-3 w-full md:w-auto justify-end">
                  <Filter className="w-4 h-4 text-zinc-500" />
                  <span className="text-xs font-bold text-zinc-400 uppercase tracking-wider">Sort:</span>
                  <select
                    id="zip-sort-select"
                    value={sortBy}
                    onChange={(e: any) => setSortBy(e.target.value)}
                    className="bg-black/60 border border-white/10 text-xs text-white rounded-xl px-3 py-2 focus:outline-none focus:border-white/30 font-bold uppercase tracking-wider"
                  >
                    <option value="confidence_desc">Highest Confidence</option>
                    <option value="confidence_asc">Lowest Confidence</option>
                    <option value="name">File Name</option>
                  </select>
                </div>
              </div>


              {/* ADANIMATION STAGGERED VISUAL MATRIX GRID */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {getFilteredZipPredictions().map((item, index) => {
                  const isLowConf = item.predicted_class === "Unknown / Low Confidence" || item.confidence < 0.6;
                  
                  return (
                    <div
                      key={index}
                      style={{ animationDelay: `${index * 80}ms` }}
                      className="glass-panel rounded-3xl border border-white/10 hover:border-white/30 bg-black/40 overflow-hidden group relative flex flex-col transition-all duration-500 animate-in fade-in zoom-in-95"
                    >
                      {/* Image Thumbnail Container with Cyber Scanner Hover Reticle */}
                      <div className="h-48 bg-zinc-950 relative overflow-hidden flex items-center justify-center border-b border-white/10 group-hover:border-white/20">
                        {item.image_data_url ? (
                          <img
                            src={item.image_data_url}
                            alt={item.filename}
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                          />
                        ) : (
                          <div className="flex flex-col items-center justify-center text-zinc-600">
                            <ImageIcon className="w-10 h-10 mb-2" />
                            <span className="text-[10px] uppercase font-mono">No Preview</span>
                          </div>
                        )}

                        {/* Scanner Laser Sweep on Hover */}
                        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/5 to-transparent -translate-y-full group-hover:translate-y-full transition-transform duration-1000 pointer-events-none" />

                        {/* Expand Modal Action Button Overlay */}
                        <button
                          onClick={() => setInspectedItem(item)}
                          className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 backdrop-blur-xs transition-opacity duration-300 flex items-center justify-center text-white font-extrabold text-xs uppercase tracking-widest gap-2"
                        >
                          <Maximize2 className="w-4 h-4 text-amber-400" />
                          Inspect Telemetry
                        </button>

                        {/* Class Tag Overlay */}
                        <div className="absolute top-3 left-3 z-10">
                          <span className={`px-3 py-1 rounded-xl text-[10px] font-extrabold tracking-widest uppercase border backdrop-blur-md shadow-md ${
                            isLowConf
                              ? 'bg-zinc-900/90 text-zinc-400 border-zinc-700'
                              : 'bg-black/80 text-white border-white/20'
                          }`}>
                            {item.predicted_class}
                          </span>
                        </div>
                      </div>

                      {/* Card Content & Confidence Bar */}
                      <div className="p-5 flex-1 flex flex-col justify-between space-y-4">
                        <div>
                          <p className="text-xs font-bold text-white tracking-wide truncate font-mono" title={item.filename}>
                            {item.filename.split('/').pop()}
                          </p>
                          <p className="text-[10px] text-zinc-500 font-mono tracking-widest mt-0.5 truncate">
                            Path: {item.filename}
                          </p>
                        </div>

                        <div>
                          <div className="flex justify-between items-center text-[10px] font-bold tracking-widest uppercase mb-1.5">
                            <span className="text-zinc-500">Confidence</span>
                            <span className={isLowConf ? 'text-zinc-500' : 'text-amber-400 font-mono'}>
                              {(item.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-2 w-full bg-black/80 rounded-full overflow-hidden border border-white/10">
                            <div
                              className={`h-full transition-all duration-700 ${
                                isLowConf
                                  ? 'bg-zinc-600'
                                  : 'bg-gradient-to-r from-amber-500 to-amber-300 shadow-[0_0_10px_rgba(245,158,11,0.5)]'
                              }`}
                              style={{ width: `${item.confidence * 100}%` }}
                            />
                          </div>
                        </div>

                        <div className="flex items-center justify-between pt-2 border-t border-white/5 text-[10px] text-zinc-500 font-mono">
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {item.inference_time_ms.toFixed(1)} ms
                          </span>
                          <button
                            onClick={() => setInspectedItem(item)}
                            className="text-amber-400 hover:text-amber-300 font-extrabold uppercase tracking-wider"
                          >
                            Details &rarr;
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {getFilteredZipPredictions().length === 0 && (
                <div className="text-center py-16 border border-dashed border-white/10 rounded-3xl bg-black/20 text-zinc-500">
                  <Search className="w-10 h-10 mx-auto mb-3 opacity-30" />
                  <p className="font-extrabold uppercase tracking-widest text-sm">No items matching criteria</p>
                  <button onClick={() => { setSearchQuery(''); setFilterClass('ALL'); }} className="mt-3 text-xs text-amber-400 hover:underline">
                    Reset search filters
                  </button>
                </div>
              )}

            </div>
          )}

        </div>
      )}


      {/* ========================================================================= */}
      {/* OPERATION: STUDENT PLACEMENT PREDICTOR FORM */}
      {/* ========================================================================= */}
      {projectMode === 'placement' && (
        <div className="space-y-8 animate-in fade-in duration-500">

          {/* ── Predefined Student Profiles ── */}
          <div className="glass-panel p-6 rounded-3xl border border-white/10 relative overflow-hidden">
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-2 text-xs font-extrabold tracking-[0.2em] uppercase text-zinc-300">
                <Sparkles className="w-4 h-4 text-amber-400 animate-pulse" />
                Predefined Student Profiles — Click to Apply &amp; Analyse
              </div>
              <span className="text-[10px] bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 px-3 py-1 rounded-full font-bold uppercase tracking-widest flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping inline-block" />
                Instant Inference
              </span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                {
                  label: 'Top Performer', icon: '🏆',
                  color: 'emerald',
                  desc: 'High achiever, placed candidate profile',
                  vals: { cgpa: 9.2, aptitude: 92, comm: 5, coding: 5, internship: 'Yes' as 'Yes'|'No', projects: 5 }
                },
                {
                  label: 'Strong Candidate', icon: '⭐',
                  color: 'blue',
                  desc: 'Above-average with good skills',
                  vals: { cgpa: 8.1, aptitude: 78, comm: 4, coding: 4, internship: 'Yes' as 'Yes'|'No', projects: 3 }
                },
                {
                  label: 'Average Student', icon: '📘',
                  color: 'amber',
                  desc: 'Typical mid-tier placement prospect',
                  vals: { cgpa: 7.0, aptitude: 65, comm: 3, coding: 3, internship: 'No' as 'Yes'|'No', projects: 2 }
                },
                {
                  label: 'At-Risk Profile', icon: '⚠️',
                  color: 'rose',
                  desc: 'Low scores, unlikely placement outcome',
                  vals: { cgpa: 5.8, aptitude: 48, comm: 2, coding: 2, internship: 'No' as 'Yes'|'No', projects: 0 }
                },
              ].map((profile) => (
                <button
                  key={profile.label}
                  disabled={tabularPredicting}
                  onClick={async () => {
                    setCgpa(profile.vals.cgpa);
                    setAptitudeScore(profile.vals.aptitude);
                    setCommunicationSkills(profile.vals.comm);
                    setCodingSkills(profile.vals.coding);
                    setInternship(profile.vals.internship);
                    setProjectsCompleted(profile.vals.projects);
                    setTabularResult(null);
                    // Immediately run inference with the profile values
                    setTabularPredicting(true);
                    try {
                      const res = await axios.post(`${apiUrl}/predict/tabular`, {
                        cgpa: profile.vals.cgpa,
                        aptitude_score: profile.vals.aptitude,
                        communication_skills: profile.vals.comm,
                        coding_skills: profile.vals.coding,
                        internship: profile.vals.internship,
                        projects_completed: profile.vals.projects
                      });
                      setTabularResult(res.data);
                    } catch (e: any) {
                      alert(e.response?.data?.detail || 'Tabular prediction failed');
                    } finally {
                      setTabularPredicting(false);
                    }
                  }}
                  className={`flex flex-col items-start gap-3 p-5 bg-black/40 hover:bg-white/5 border rounded-2xl transition-all duration-300 text-left group relative overflow-hidden ${
                    profile.color === 'emerald' ? 'border-emerald-500/20 hover:border-emerald-400/40' :
                    profile.color === 'blue' ? 'border-blue-500/20 hover:border-blue-400/40' :
                    profile.color === 'amber' ? 'border-amber-500/20 hover:border-amber-400/40' :
                    'border-rose-500/20 hover:border-rose-400/40'
                  }`}
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-transparent via-white/[0.02] to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
                  <div className="flex items-center gap-2 w-full">
                    <span className="text-2xl">{profile.icon}</span>
                    <span className="font-extrabold text-white text-sm tracking-wide group-hover:text-amber-300 transition-colors flex-1">{profile.label}</span>
                    <Play className="w-3.5 h-3.5 text-amber-500/60 group-hover:text-amber-400 fill-current transition-colors" />
                  </div>
                  <p className="text-[10px] text-zinc-500 font-light leading-relaxed">{profile.desc}</p>
                  <div className="flex flex-wrap gap-1.5 mt-1">
                    <span className="text-[9px] bg-white/5 border border-white/10 px-2 py-0.5 rounded font-mono text-zinc-400">CGPA {profile.vals.cgpa}</span>
                    <span className="text-[9px] bg-white/5 border border-white/10 px-2 py-0.5 rounded font-mono text-zinc-400">APT {profile.vals.aptitude}</span>
                    <span className={`text-[9px] px-2 py-0.5 rounded font-mono border ${
                      profile.vals.internship === 'Yes' 
                        ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' 
                        : 'bg-zinc-800/50 border-white/10 text-zinc-500'
                    }`}>{profile.vals.internship === 'Yes' ? '✓ Internship' : '✗ No Internship'}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
          {/* Parameters Form */}
          <div className="glass-panel rounded-3xl p-8 shadow-2xl relative overflow-hidden flex flex-col group transition-all duration-300">
            <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />
            
            <h2 className="text-xl font-bold mb-6 flex items-center gap-3 uppercase tracking-widest text-zinc-100 relative z-10">
              <Sliders className="w-6 h-6 text-white" />
              Student Parameters
            </h2>

            <div className="space-y-6 relative z-10 flex-grow">
              {/* CGPA */}
              <div>
                <div className="flex justify-between text-xs font-bold tracking-[0.1em] uppercase mb-2">
                  <span className="text-zinc-400">CGPA (Cumulative Grade Point Average)</span>
                  <span className="text-white font-mono font-bold">{cgpa.toFixed(2)} / 10.0</span>
                </div>
                <input 
                  type="range" min="5.0" max="10.0" step="0.1"
                  value={cgpa} 
                  onChange={e => setCgpa(parseFloat(e.target.value))}
                  className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-white"
                />
              </div>

              {/* Aptitude Score */}
              <div>
                <div className="flex justify-between text-xs font-bold tracking-[0.1em] uppercase mb-2">
                  <span className="text-zinc-400">Aptitude Test Score</span>
                  <span className="text-white font-mono font-bold">{aptitudeScore} / 100</span>
                </div>
                <input 
                  type="range" min="40" max="100" step="1"
                  value={aptitudeScore} 
                  onChange={e => setAptitudeScore(parseInt(e.target.value))}
                  className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-white"
                />
              </div>

              {/* Grid for Comm & Coding Skills */}
              <div className="grid grid-cols-2 gap-6">
                {/* Communication Skills */}
                <div>
                  <div className="flex justify-between text-[10px] font-bold tracking-[0.1em] uppercase mb-2">
                    <span className="text-zinc-400">Communication Skills</span>
                    <span className="text-white font-mono font-bold">{communicationSkills} / 5</span>
                  </div>
                  <input 
                    type="range" min="1" max="5" step="1"
                    value={communicationSkills} 
                    onChange={e => setCommunicationSkills(parseInt(e.target.value))}
                    className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-white"
                  />
                </div>

                {/* Coding Skills */}
                <div>
                  <div className="flex justify-between text-[10px] font-bold tracking-[0.1em] uppercase mb-2">
                    <span className="text-zinc-400">Coding Skills</span>
                    <span className="text-white font-mono font-bold">{codingSkills} / 5</span>
                  </div>
                  <input 
                    type="range" min="1" max="5" step="1"
                    value={codingSkills} 
                    onChange={e => setCodingSkills(parseInt(e.target.value))}
                    className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-white"
                  />
                </div>
              </div>

              {/* Grid for Internship & Projects */}
              <div className="grid grid-cols-2 gap-6">
                {/* Internship */}
                <div>
                  <label className="block text-xs font-bold text-zinc-400 mb-2 tracking-[0.1em] uppercase">Internship Experience</label>
                  <select 
                    value={internship} 
                    onChange={e => setInternship(e.target.value as 'Yes' | 'No')}
                    className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-white/30 transition-colors appearance-none cursor-pointer uppercase font-bold text-xs tracking-wider"
                  >
                    <option value="No">No (None)</option>
                    <option value="Yes">Yes (Completed)</option>
                  </select>
                </div>

                {/* Projects Completed */}
                <div>
                  <div className="flex justify-between text-xs font-bold tracking-[0.1em] uppercase mb-2">
                    <span className="text-zinc-400">Projects Completed</span>
                    <span className="text-white font-mono font-bold">{projectsCompleted}</span>
                  </div>
                  <input 
                    type="number" min="0" max="10"
                    value={projectsCompleted} 
                    onChange={e => setProjectsCompleted(parseInt(e.target.value) || 0)}
                    className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-white/30 transition-colors font-mono font-bold"
                  />
                </div>
              </div>
            </div>

            <button 
              onClick={handlePredictTabular}
              disabled={tabularPredicting}
              className="w-full mt-8 flex items-center justify-center gap-3 bg-white hover:bg-zinc-200 disabled:bg-white/5 disabled:text-zinc-500 text-black font-extrabold py-4 px-4 rounded-xl transition-all shadow-[0_0_20px_rgba(255,255,255,0.1)] tracking-[0.1em] uppercase text-sm relative z-10"
            >
              {tabularPredicting ? (
                <><Loader2 className="w-5 h-5 animate-spin" /> Forwarding Tensors...</>
              ) : (
                <><Play className="w-5 h-5 fill-current" /> Run ANN Inference</>
              )}
            </button>
          </div>

          {/* Inference Visualization */}
          <div className="glass-panel rounded-3xl p-8 shadow-2xl relative overflow-hidden group transition-all duration-300 min-h-[500px] flex flex-col justify-between">
            <div className="absolute bottom-0 left-0 w-64 h-64 bg-white/5 rounded-full blur-3xl translate-y-1/2 -translate-x-1/4 pointer-events-none" />
            <h2 className="text-xl font-bold mb-6 flex items-center gap-3 uppercase tracking-widest text-zinc-100 relative z-10">
              <CheckCircle className="w-6 h-6 text-white" />
              Inference Outcome
            </h2>

            {tabularPredicting ? (
              <div className="flex-grow flex flex-col items-center justify-center text-center py-20 relative z-10">
                <Loader2 className="w-16 h-16 text-white animate-spin mb-4" />
                <p className="font-mono text-xs text-zinc-400 uppercase tracking-widest animate-pulse">Running forward pass...</p>
              </div>
            ) : tabularResult ? (
              <div className="space-y-8 animate-in slide-in-from-right-8 duration-500 relative z-10 flex-grow flex flex-col justify-between">
                <div className="text-center p-8 bg-black/40 border border-white/5 rounded-3xl shadow-inner backdrop-blur-sm relative overflow-hidden">
                  <div className="absolute inset-0 bg-radial-gradient from-emerald-500/10 via-transparent to-transparent pointer-events-none opacity-50" />
                  
                  <p className="text-xs text-zinc-400 font-bold tracking-[0.2em] uppercase mb-4">PREDICTION MATRIX</p>
                  
                  <div className="flex justify-center mb-6">
                    <div className={`relative w-40 h-40 rounded-full flex flex-col items-center justify-center border-4 ${
                      tabularResult.prediction === 'Placed' 
                        ? 'border-emerald-500/35 shadow-[0_0_30px_rgba(16,185,129,0.2)] bg-emerald-500/5' 
                        : 'border-rose-500/35 shadow-[0_0_30px_rgba(239,68,68,0.2)] bg-rose-500/5'
                    }`}>
                      <span className={`text-4xl font-black ${tabularResult.prediction === 'Placed' ? 'text-emerald-400' : 'text-rose-400'}`}>
                        {tabularResult.prediction === 'Placed' ? 'YES' : 'NO'}
                      </span>
                      <span className="text-[10px] text-zinc-500 font-bold uppercase mt-1">PLACED</span>
                    </div>
                  </div>

                  <h3 className={`text-3xl font-extrabold tracking-tight ${
                    tabularResult.prediction === 'Placed' ? 'text-emerald-400' : 'text-rose-400'
                  }`}>
                    {tabularResult.prediction === 'Placed' ? 'Placement Likely' : 'Placement Unlikely'}
                  </h3>
                  
                  <div className="mt-8 max-w-sm mx-auto">
                    <div className="flex justify-between text-xs font-bold tracking-widest uppercase mb-3">
                      <span className="text-zinc-500">Placement Probability</span>
                      <span className="text-white">{(tabularResult.placement_probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-4 w-full bg-black/60 rounded-full overflow-hidden border border-white/10 shadow-inner">
                      <div 
                        className={`h-full transition-all duration-1000 ${
                          tabularResult.prediction === 'Placed' ? 'bg-emerald-500 shadow-[0_0_20px_#10b981]' : 'bg-rose-500 shadow-[0_0_20px_#ef4444]'
                        }`}
                        style={{ width: `${tabularResult.placement_probability * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-6 pt-6 border-t border-white/10">
                  <div className="flex items-center gap-3 text-xs font-bold tracking-widest uppercase text-zinc-500">
                    <Clock className="w-5 h-5 text-zinc-400" />
                    Latency: <span className="font-mono text-white ml-1">{tabularResult.inference_time_ms.toFixed(1)} ms</span>
                  </div>
                  <div className="text-zinc-800 text-xl font-light">|</div>
                  <div className="text-xs font-bold tracking-widest uppercase text-zinc-500">
                    Model: <span className="text-white ml-2">{activeModel.name}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-zinc-600 min-h-[350px] relative z-10 flex-grow">
                <TargetPlaceholder />
                <p className="mt-8 font-light tracking-[0.2em] uppercase text-sm">Awaiting student telemetry input...</p>
              </div>
            )}
          </div>
        </div>
        </div>
      )}


      {/* ========================================================================= */}
      {/* DETAILED ITEM INSPECTION MODAL */}
      {/* ========================================================================= */}
      {inspectedItem && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4 animate-in fade-in duration-200">
          <div className="glass-panel border border-white/20 rounded-3xl max-w-2xl w-full p-8 relative overflow-hidden shadow-2xl space-y-6">
            
            {/* Modal Header */}
            <div className="flex items-start justify-between border-b border-white/10 pb-4">
              <div>
                <span className="text-[10px] font-extrabold tracking-[0.2em] text-amber-400 uppercase">Telemetry Item Inspection</span>
                <h3 className="text-xl font-extrabold text-white tracking-wide font-mono mt-1" title={inspectedItem.filename}>
                  {inspectedItem.filename.split('/').pop()}
                </h3>
              </div>
              <button
                onClick={() => setInspectedItem(null)}
                className="p-2 bg-white/5 hover:bg-white/10 border border-white/10 text-zinc-400 hover:text-white rounded-xl transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              {/* Preview Image */}
              <div className="h-64 bg-zinc-950 rounded-2xl border border-white/10 overflow-hidden flex items-center justify-center relative">
                {inspectedItem.image_data_url ? (
                  <img src={inspectedItem.image_data_url} alt={inspectedItem.filename} className="w-full h-full object-contain p-2" />
                ) : (
                  <ImageIcon className="w-12 h-12 text-zinc-700" />
                )}
              </div>

              {/* Class Probabilities */}
              <div className="space-y-4 flex flex-col justify-between">
                <div>
                  <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1">Target Classification</p>
                  <h4 className="text-3xl font-extrabold text-white tracking-tight">{inspectedItem.predicted_class}</h4>
                  <p className="text-xs text-amber-400 font-mono font-bold mt-1">
                    Confidence: {(inspectedItem.confidence * 100).toFixed(2)}%
                  </p>
                </div>

                <div className="space-y-2">
                  <p className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Top Probability Candidates</p>
                  {inspectedItem.top_3.map((cand, idx) => (
                    <div key={idx} className="bg-black/50 border border-white/5 p-3 rounded-xl space-y-1">
                      <div className="flex justify-between text-xs font-bold">
                        <span className="text-zinc-300">{cand.class}</span>
                        <span className="text-white font-mono font-bold">{(cand.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                        <div className="h-full bg-amber-400 rounded-full" style={{ width: `${cand.confidence * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>

                <div className="flex items-center gap-4 text-xs text-zinc-500 font-mono pt-2 border-t border-white/5">
                  <span>Latency: <strong className="text-white">{inspectedItem.inference_time_ms.toFixed(1)} ms</strong></span>
                  <span>|</span>
                  <span>Path: <strong className="text-zinc-300">{inspectedItem.filename}</strong></span>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="pt-2 text-right">
              <button
                onClick={() => setInspectedItem(null)}
                className="bg-white hover:bg-zinc-200 text-black font-extrabold px-6 py-2.5 rounded-xl text-xs uppercase tracking-widest transition-colors"
              >
                Close Inspection
              </button>
            </div>

          </div>
        </div>
      )}

    </div>
  );
}

function TargetPlaceholder() {
  return (
    <svg className="w-32 h-32 opacity-10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="0.5">
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="12" r="6" />
      <circle cx="12" cy="12" r="2" />
      <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
    </svg>
  );
}
