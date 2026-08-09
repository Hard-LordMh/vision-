import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Upload, Database, Loader2, Image as ImageIcon, Download, Sparkles, FileArchive, Zap } from 'lucide-react';

import { getApiUrl } from '../config';

export default function Datasets() {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isPreloading, setIsPreloading] = useState(false);
  const [isPreloadingKaggle, setIsPreloadingKaggle] = useState(false);
  const [isLoadingHuman, setIsLoadingHuman] = useState(false);
  const [uploadName, setUploadName] = useState('');

  const [apiUrl, setApiUrl] = useState(getApiUrl());
  const [projectMode, setProjectMode] = useState<'vision' | 'placement'>(() => {
    return (localStorage.getItem('visionai_project_mode') as 'vision' | 'placement') || 'vision';
  });

  const fetchDatasets = async (customApi?: string) => {
    try {
      const api = customApi || getApiUrl();
      const res = await axios.get(`${api}/datasets`);
      setDatasets(res.data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    const activeApi = getApiUrl();
    setApiUrl(activeApi);
    fetchDatasets(activeApi);

    const handleModeChange = () => {
      setProjectMode((localStorage.getItem('visionai_project_mode') as 'vision' | 'placement') || 'vision');
    };
    window.addEventListener('projectModeChanged', handleModeChange);
    return () => window.removeEventListener('projectModeChanged', handleModeChange);
  }, []);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const allowed = projectMode === 'vision' ? file.name.endsWith('.zip') : file.name.endsWith('.csv');
    if (!allowed) {
      alert(projectMode === 'vision' ? "Only .zip files are supported for Vision AI." : "Only .csv files are supported for Placement prediction.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", uploadName || file.name.replace(/\.[^/.]+$/, ""));

    setIsUploading(true);
    try {
      await axios.post(`${apiUrl}/datasets/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUploadName('');
      fetchDatasets(apiUrl);
    } catch (e) {
      console.error(e);
      alert("Upload failed. Make sure it's a valid data file.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadPreloadedDataset = async () => {
    setIsPreloading(true);
    try {
      const endpoint = projectMode === 'vision' ? 'load-dataset' : 'load-placement-dataset';
      await axios.post(`${apiUrl}/samples/${endpoint}`);
      fetchDatasets(apiUrl);
    } catch (e) {
      console.error(e);
      alert("Failed to load preloaded dataset");
    } finally {
      setIsPreloading(false);
    }
  };

  const handleLoadKaggleDataset = async () => {
    setIsPreloadingKaggle(true);
    try {
      await axios.post(`${apiUrl}/samples/load-real-placement-dataset`);
      fetchDatasets(apiUrl);
    } catch (e) {
      console.error(e);
      alert("Failed to load Kaggle dataset");
    } finally {
      setIsPreloadingKaggle(false);
    }
  };

  const handleLoadHumanDataset = async () => {
    setIsLoadingHuman(true);
    try {
      await axios.post(`${apiUrl}/samples/load-human-dataset`);
      fetchDatasets(apiUrl);
      alert('✅ Human Detection Model trained & activated! You can now predict human images.');
    } catch (e: any) {
      console.error(e);
      alert(e.response?.data?.detail || "Failed to load human detection dataset");
    } finally {
      setIsLoadingHuman(false);
    }
  };

  const navigateToZipPredict = (autoLoadPreloaded: boolean = false) => {
    navigate('/predict', { state: { mode: 'zip', autoLoadPreloaded } });
  };

  const filteredDatasets = datasets.filter((ds: any) => {
    const isPlacement = projectMode === 'placement';
    const hasPlacementName = ds.name.toLowerCase().includes('placement');
    return isPlacement ? hasPlacementName : !hasPlacementName;
  });

  return (
    <div className="space-y-10 animate-in fade-in slide-in-from-bottom-8 duration-700">
      <header className="relative">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-2 uppercase">
          Data Matrix<span className="text-zinc-500">.</span>
        </h1>
        <p className="text-zinc-400 text-lg tracking-wide max-w-xl font-light">
          Ingest and structure raw data blocks for neural processing.
        </p>
      </header>

      {/* Preloaded Dataset Utilities */}
      <div className="glass-panel p-6 rounded-3xl border border-white/10 relative overflow-hidden flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-2xl text-amber-400 shrink-0">
            <Sparkles className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-lg font-extrabold text-white uppercase tracking-wider">
              {projectMode === 'vision' ? 'Vision Dataset Library' : 'Preloaded Placement Data Block'}
            </h3>
            <p className="text-xs text-zinc-400 font-light mt-0.5">
              {projectMode === 'vision' 
                ? 'Import Animals Dataset (Cat, Dog, Bird) · Or load the Human Detection Model to classify humans + animals (4 classes) using a fine-tuned pretrained backbone.'
                : 'Pre-configured Student Placement Dataset (500 records) with CGPA, Aptitude, Comm/Coding skills, Internship, and Outcomes.'}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3 w-full md:w-auto">
          {projectMode === 'vision' ? (
            <>
              <button
                onClick={handleLoadPreloadedDataset}
                disabled={isPreloading || isLoadingHuman}
                className="flex-grow md:flex-initial flex items-center justify-center gap-2 bg-white hover:bg-zinc-200 text-black font-extrabold text-xs px-5 py-3 rounded-xl uppercase tracking-widest transition-all shadow-md disabled:opacity-50"
              >
                {isPreloading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Database className="w-4 h-4" />}
                {isPreloading ? "Ingesting..." : "Import Animals Dataset"}
              </button>

              <button
                onClick={handleLoadHumanDataset}
                disabled={isPreloading || isLoadingHuman}
                title="Creates a 4-class (Human, Cat, Dog, Bird) dataset and trains MobileNetV3 using pretrained ImageNet backbone. Takes ~1-3 min."
                className={`flex-grow md:flex-initial flex items-center justify-center gap-2 font-extrabold text-xs px-5 py-3 rounded-xl uppercase tracking-widest transition-all shadow-md disabled:opacity-50 ${
                  isLoadingHuman 
                    ? 'bg-purple-500/30 border border-purple-500/50 text-purple-300' 
                    : 'bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white shadow-[0_0_20px_rgba(139,92,246,0.3)]'
                }`}
              >
                {isLoadingHuman ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                {isLoadingHuman ? 'Training Human Model...' : '🧑 Load Human Detection Model'}
              </button>

              <button
                onClick={() => navigateToZipPredict(true)}
                className="flex-grow md:flex-initial flex items-center justify-center gap-2 bg-amber-400 hover:bg-amber-300 text-black font-extrabold text-xs px-5 py-3 rounded-xl uppercase tracking-widest transition-all shadow-[0_0_15px_rgba(245,158,11,0.3)]"
              >
                <Zap className="w-4 h-4 fill-current" />
                Predict ZIP Matrix
              </button>

              <a
                href={`${apiUrl.replace('/api', '')}/static/samples/sample_animals_dataset.zip`}
                download
                className="flex-grow md:flex-initial flex items-center justify-center gap-2 bg-white/10 hover:bg-white/20 border border-white/20 text-white font-bold text-xs px-4 py-3 rounded-xl uppercase tracking-widest transition-all"
              >
                <Download className="w-4 h-4" />
                Download ZIP
              </a>
            </>
          ) : (
            <>
              <button
                onClick={handleLoadPreloadedDataset}
                disabled={isPreloading || isPreloadingKaggle}
                className="flex-grow md:flex-initial flex items-center justify-center gap-2 bg-white/10 hover:bg-white/20 border border-white/20 text-white font-extrabold text-xs px-5 py-3 rounded-xl uppercase tracking-widest transition-all disabled:opacity-50"
              >
                {isPreloading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Database className="w-4 h-4" />}
                {isPreloading ? "Importing Synthetic..." : "Import Synthetic Data"}
              </button>

              <button
                onClick={handleLoadKaggleDataset}
                disabled={isPreloading || isPreloadingKaggle}
                className="flex-grow md:flex-initial flex items-center justify-center gap-2 bg-white hover:bg-zinc-200 text-black font-extrabold text-xs px-5 py-3 rounded-xl uppercase tracking-widest transition-all shadow-md disabled:opacity-50"
              >
                {isPreloadingKaggle ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                {isPreloadingKaggle ? "Importing Kaggle..." : "Import Kaggle Data"}
              </button>
            </>
          )}
        </div>
      </div>

      {/* Upload Zone */}
      <div className="glass-panel rounded-3xl p-8 shadow-2xl relative overflow-hidden group max-w-3xl">
        <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />
        <h2 className="text-xl font-bold mb-8 flex items-center gap-3 uppercase tracking-widest text-zinc-100">
          <Upload className="w-6 h-6 text-white" />
          Initialize Upload Sequence
        </h2>
        
        <div className="space-y-6 relative z-10">
          <div>
            <label className="block text-xs font-bold text-zinc-400 mb-2 tracking-[0.1em] uppercase">Dataset Designation</label>
            <input 
              type="text" 
              value={uploadName}
              onChange={(e) => setUploadName(e.target.value)}
              placeholder="e.g. PROJECT_OMEGA_V1" 
              className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-white/30 transition-colors placeholder:text-zinc-700"
            />
          </div>
          
          <div className="relative border-2 border-dashed border-white/10 hover:border-white/30 rounded-2xl p-10 text-center transition-all duration-300 bg-white/[0.02] group-hover:bg-white/[0.04] shadow-inner">
            <input 
              type="file" 
              accept={projectMode === 'vision' ? '.zip' : '.csv'} 
              onChange={handleUpload}
              disabled={isUploading}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
            />
            {isUploading ? (
              <div className="flex flex-col items-center text-white">
                <Loader2 className="w-10 h-10 animate-spin mb-4 drop-shadow-[0_0_10px_rgba(255,255,255,0.2)]" />
                <p className="font-bold tracking-widest uppercase text-sm">Parsing Archive Block...</p>
              </div>
            ) : (
              <div className="flex flex-col items-center text-zinc-500 group-hover:text-white transition-colors duration-300">
                <Database className="w-12 h-12 mb-4 drop-shadow-[0_0_15px_rgba(255,255,255,0)] group-hover:drop-shadow-[0_0_15px_rgba(255,255,255,0.1)] transition-all" />
                <p className="font-bold tracking-widest uppercase text-sm text-zinc-300 group-hover:text-white">
                  {projectMode === 'vision' ? 'Drop ZIP Archive to Upload Dataset' : 'Drop CSV File to Upload Tabular Dataset'}
                </p>
                <p className="text-xs mt-2 font-light tracking-wide text-zinc-600 group-hover:text-zinc-400">
                  {projectMode === 'vision' ? 'Format: Standard ZIP classification structure' : 'Format: Comma-Separated Values student dataset (.CSV)'}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Dataset Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredDatasets.map((ds: any) => (
          <div key={ds.id} className="glass-panel rounded-3xl p-6 relative overflow-hidden group transition-all duration-500 flex flex-col justify-between">
            <div className="absolute top-0 right-0 w-32 h-32 bg-white/5 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2 pointer-events-none group-hover:bg-white/10 transition-colors" />
            
            <div>
              <div className="flex items-start justify-between mb-6 relative z-10">
                <div>
                  <h3 className="text-xl font-extrabold tracking-tight text-zinc-100 group-hover:text-white transition-colors">{ds.name}</h3>
                  <span className="text-xs text-zinc-500 font-bold tracking-wider uppercase mt-1 block">Origin: {ds.source}</span>
                </div>
                <div className="bg-white/5 border border-white/10 text-white px-3 py-1.5 rounded-lg text-xs font-bold tracking-widest shadow-[0_0_15px_rgba(255,255,255,0.05)]">
                  {ds.total_images.toLocaleString()} {projectMode === 'vision' ? 'IMG' : 'ROWS'}
                </div>
              </div>
              
              <div className="flex items-center gap-3 text-sm text-zinc-400 mb-6 font-bold tracking-wide uppercase relative z-10">
                <ImageIcon className="w-4 h-4 text-zinc-400" />
                <span>{ds.class_count} {projectMode === 'vision' ? 'Categories' : 'Outcomes'}</span>
              </div>
              
              <div className="space-y-2 relative z-10 mb-6">
                <p className="text-[10px] font-bold text-zinc-600 uppercase tracking-[0.2em] mb-2">Registry Entries</p>
                <div className="flex flex-wrap gap-2">
                  {ds.classes.slice(0, 5).map((c: any) => (
                    <span key={c.id} className="bg-white/5 border border-white/10 text-zinc-300 px-2 py-1 rounded-md text-xs font-medium tracking-wide hover:border-white/30 transition-colors">
                      {c.name}
                    </span>
                  ))}
                  {ds.classes.length > 5 && (
                    <span className="bg-black border border-white/10 text-zinc-400 px-2 py-1 rounded-md text-xs font-bold">
                      +{ds.classes.length - 5}
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="pt-4 border-t border-white/10 flex items-center justify-between relative z-10">
              {projectMode === 'vision' ? (
                <button
                  onClick={() => navigateToZipPredict(true)}
                  className="w-full flex items-center justify-center gap-2 bg-white/5 hover:bg-amber-400 hover:text-black text-zinc-300 font-extrabold text-xs py-2.5 rounded-xl uppercase tracking-widest transition-all border border-white/10 hover:border-amber-400"
                >
                  <FileArchive className="w-4 h-4" />
                  Batch Predict ZIP
                </button>
              ) : (
                <button
                  onClick={() => navigate('/predict')}
                  className="w-full flex items-center justify-center gap-2 bg-white/5 hover:bg-white hover:text-black text-zinc-300 font-extrabold text-xs py-2.5 rounded-xl uppercase tracking-widest transition-all border border-white/10 hover:border-white"
                >
                  <Zap className="w-4 h-4" />
                  Open Predictor Form
                </button>
              )}
            </div>

          </div>
        ))}
        {filteredDatasets.length === 0 && (
          <div className="col-span-full text-center py-16 text-zinc-600 border border-dashed border-white/10 rounded-3xl bg-white/[0.01]">
            <Database className="w-12 h-12 mx-auto mb-4 opacity-20" />
            <p className="font-bold tracking-widest uppercase text-sm">No operational datablocks detected.</p>
          </div>
        )}
      </div>
    </div>
  );
}
