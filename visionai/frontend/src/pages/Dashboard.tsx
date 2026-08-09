import { useState, useEffect } from 'react';
import axios from 'axios';
import { Activity, Database, Box, Target } from 'lucide-react';
import { Link } from 'react-router-dom';

import { getApiUrl } from '../config';

export default function Dashboard() {
  const [projectMode, setProjectMode] = useState<'vision' | 'placement'>(() => {
    return (localStorage.getItem('visionai_project_mode') as 'vision' | 'placement') || 'vision';
  });

  const [rawData, setRawData] = useState<{datasets: any[], runs: any[], models: any[]}>({
    datasets: [],
    runs: [],
    models: []
  });

  useEffect(() => {
    const handleModeChange = () => {
      setProjectMode((localStorage.getItem('visionai_project_mode') as 'vision' | 'placement') || 'vision');
    };
    window.addEventListener('projectModeChanged', handleModeChange);
    return () => window.removeEventListener('projectModeChanged', handleModeChange);
  }, []);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const apiUrl = getApiUrl();
        const [dsRes, trRes, mdRes] = await Promise.all([
          axios.get(`${apiUrl}/datasets`),
          axios.get(`${apiUrl}/training`),
          axios.get(`${apiUrl}/models`)
        ]);
        setRawData({
          datasets: dsRes.data,
          runs: trRes.data,
          models: mdRes.data
        });
      } catch (e) {
        console.error("Failed to load stats", e);
      }
    };
    fetchStats();
  }, []);

  const getFilteredStats = () => {
    const isPlacement = projectMode === 'placement';
    
    const filteredDs = rawData.datasets.filter(d => 
      isPlacement ? d.name.toLowerCase().includes('placement') : !d.name.toLowerCase().includes('placement')
    );
    
    const filteredRuns = rawData.runs.filter(r => 
      isPlacement ? r.model_architecture.startsWith('ANN') : !r.model_architecture.startsWith('ANN')
    );
    
    const filteredModels = rawData.models.filter(m => 
      isPlacement ? m.architecture.startsWith('ANN') : !m.architecture.startsWith('ANN')
    );
    
    let bestAcc = 0;
    filteredModels.forEach((m: any) => {
      if (m.accuracy > bestAcc) bestAcc = m.accuracy;
    });

    return {
      datasets: filteredDs.length,
      runs: filteredRuns.length,
      models: filteredModels.length,
      bestAcc: bestAcc
    };
  };

  const stats = getFilteredStats();

  return (
    <div className="space-y-8 animate-mac-scale">
      <header className="relative pb-4 border-b border-white/5">
        <h1 className="text-3xl font-extrabold tracking-tight text-white mb-1.5 uppercase">
          {projectMode === 'vision' ? 'System Overview' : 'Placement Matrix Overview'}
        </h1>
        <p className="text-zinc-400 text-xs tracking-wider uppercase font-bold font-mono">
          {projectMode === 'vision' 
            ? 'All systems nominal. Host environment connected.' 
            : 'Predictive student placement analytics & ANN training environment.'}
        </p>
      </header>

      {/* Stats Widget Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard icon={<Database className="text-zinc-400 w-5 h-5" />} title="DATASETS" value={stats.datasets} />
        <StatCard icon={<Activity className="text-zinc-400 w-5 h-5" />} title="TRAINING RUNS" value={stats.runs} />
        <StatCard icon={<Box className="text-zinc-400 w-5 h-5" />} title="ACTIVE MODELS" value={stats.models} />
        <StatCard icon={<Target className="text-zinc-400 w-5 h-5" />} title="PEAK ACCURACY" value={`${(stats.bestAcc * 100).toFixed(1)}%`} />
      </div>

      {/* Main Panel Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Telemetry panel */}
        <div className="mac-glass rounded-2xl p-6 lg:col-span-2 relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-white/[0.01] rounded-full blur-3xl -translate-y-1/2 translate-x-1/4 pointer-events-none" />
          <h2 className="text-sm font-bold mb-4 flex items-center gap-2.5 text-zinc-300 tracking-wide uppercase">
            <Activity className="w-4 h-4 text-zinc-400" />
            {projectMode === 'vision' ? 'Neural Telemetry Feed' : 'Tabular ANN Metric Constructs'}
          </h2>
          
          {projectMode === 'vision' ? (
            <div className="flex flex-col items-center justify-center h-64 rounded-xl border border-white/5 bg-black/35 relative overflow-hidden shadow-inner">
               <div className="w-full h-1 bg-white/5 absolute top-0">
                 <div className="h-full w-1/3 bg-gradient-to-r from-transparent via-zinc-400 to-transparent animate-[progress_2s_linear_infinite]" />
               </div>
               <p className="text-zinc-500 font-mono tracking-[0.2em] text-[10px] font-bold">Awaiting telemetry payload...</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-4 p-5 h-64 rounded-xl border border-white/5 bg-black/35 relative overflow-hidden shadow-inner text-xs font-mono text-zinc-400">
               <div className="bg-white/[0.01] p-3 rounded-lg border border-white/5">
                 <div className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider mb-1.5">FEATURE CHANNELS</div>
                 <ul className="space-y-1 text-[11px]">
                   <li>• CGPA (Continuous Score)</li>
                   <li>• Aptitude Score (40-100)</li>
                   <li>• Communication (1-5 Star)</li>
                   <li>• Coding (1-5 Star)</li>
                   <li>• Internship (Yes/No)</li>
                   <li>• Projects completed (0-5)</li>
                 </ul>
               </div>
               <div className="bg-white/[0.01] p-3 rounded-lg border border-white/5 flex flex-col justify-between">
                 <div>
                   <div className="text-[10px] text-zinc-500 font-bold uppercase tracking-wider mb-1.5">TARGET CHANNEL</div>
                   <div className="text-white font-extrabold text-sm uppercase mb-1">Placement (Binary)</div>
                   <div className="text-[10px] text-zinc-500">• Placed (Yes)</div>
                   <div className="text-[10px] text-zinc-500">• Not Placed (No)</div>
                 </div>
                 <div className="pt-2 border-t border-white/5 text-[9px] text-amber-400/80 font-bold tracking-widest uppercase animate-pulse">
                   Preprocessed & Scale-Ready
                 </div>
               </div>
            </div>
          )}
        </div>

        {/* Shortcuts / Directives */}
        <div className="mac-glass rounded-2xl p-6 relative overflow-hidden flex flex-col justify-between">
          <div>
            <h2 className="text-sm font-bold mb-4 text-zinc-300 tracking-wide uppercase">Shortcuts</h2>
            <div className="space-y-2">
              <ActionLink to="/datasets" label="Initialize Dataset" desc="Upload structured training files" />
              <ActionLink to="/training" label="Execute Training" desc="Compile and run network" />
              <ActionLink to="/predict" label="Run Inference" desc="Use model for predictions" />
            </div>
          </div>
          <div className="mt-6 p-4 rounded-xl bg-white/[0.02] border border-white/5 text-[10px] text-zinc-500 text-center font-mono">
            VisionAI Core Workspace v2.5
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ icon, title, value }: { icon: React.ReactNode, title: string, value: number | string }) {
  return (
    <div className="mac-glass rounded-2xl p-5 flex flex-col justify-between relative overflow-hidden group transition-all duration-300 hover:bg-white/[0.04] hover:border-white/15">
      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-white/[0.01] to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
      
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-white/5 rounded-xl border border-white/10 group-hover:border-white/20 transition-colors duration-300">
          {icon}
        </div>
        <h3 className="text-zinc-500 text-[10px] font-bold tracking-[0.15em] uppercase">{title}</h3>
      </div>
      <p className="text-3xl font-extrabold tracking-tight text-white">{value}</p>
    </div>
  );
}

function ActionLink({ to, label, desc }: { to: string, label: string, desc: string }) {
  return (
    <Link to={to} className="block p-4 rounded-xl bg-white/[0.02] border border-white/5 hover:border-white/20 hover:bg-white/[0.04] transition-all duration-300 group shadow-inner">
      <div className="flex justify-between items-center">
        <div>
          <h4 className="text-xs font-bold text-zinc-300 tracking-wide group-hover:text-white transition-colors uppercase">{label}</h4>
          <p className="text-[10px] text-zinc-500 mt-1 font-light tracking-wide">{desc}</p>
        </div>
        <div className="w-7 h-7 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center group-hover:border-white/20 group-hover:bg-white/10 transition-all duration-300 mac-btn-spring shadow-inner">
          <span className="text-zinc-400 group-hover:text-white text-xs">→</span>
        </div>
      </div>
    </Link>
  );
}
