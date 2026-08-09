import { useState, useEffect } from 'react';
import axios from 'axios';
import { Box, Target, Calendar, CheckCircle, ShieldAlert } from 'lucide-react';

import { getApiUrl } from '../config';

export default function Models() {
  const [models, setModels] = useState([]);

  const [apiUrl, setApiUrl] = useState(getApiUrl());

  const fetchModels = async (customApi?: string) => {
    try {
      const api = customApi || getApiUrl();
      const res = await axios.get(`${api}/models`);
      setModels(res.data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    const activeApi = getApiUrl();
    setApiUrl(activeApi);
    fetchModels(activeApi);
  }, []);

  const activateModel = async (id: number) => {
    try {
      await axios.post(`${apiUrl}/models/${id}/activate`);
      fetchModels(apiUrl);
    } catch (e) {
      console.error(e);
      alert("Failed to activate model");
    }
  };

  return (
    <div className="space-y-10 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-12">
      <header className="relative">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-2 uppercase">
          Neural Registry<span className="text-zinc-500">.</span>
        </h1>
        <p className="text-zinc-400 text-lg tracking-wide max-w-xl font-light">
          Manage compiled networks and dictate operational status.
        </p>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {models.map((m: any) => (
          <div key={m.id} className={`glass-panel rounded-3xl p-8 relative overflow-hidden group transition-all duration-500 ${m.is_active ? 'border-white/30 shadow-[0_0_30px_rgba(255,255,255,0.1)]' : 'border-transparent'}`}>
            
            <div className={`absolute top-0 right-0 w-64 h-64 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none transition-colors duration-1000 ${m.is_active ? 'bg-white/10' : 'bg-white/5'}`} />

            {m.is_active && (
              <div className="absolute top-0 right-0 bg-white text-black font-extrabold text-[10px] px-4 py-2 rounded-bl-2xl tracking-[0.2em] flex items-center gap-2 uppercase shadow-[0_0_15px_rgba(255,255,255,0.2)]">
                <CheckCircle className="w-3 h-3" /> System Active
              </div>
            )}

            <div className="flex justify-between items-start mb-8 relative z-10">
              <div className="flex items-center gap-5">
                <div className={`p-4 rounded-2xl border shadow-inner ${m.is_active ? 'bg-white/10 border-white/30 text-white' : 'bg-black/50 border-white/10 text-white/50'}`}>
                  <Box className="w-8 h-8 drop-shadow-md" />
                </div>
                <div>
                  <h2 className="text-2xl font-extrabold text-white tracking-tight">{m.name}</h2>
                  <div className="flex items-center gap-3 text-zinc-400 text-xs font-bold tracking-widest mt-2 uppercase">
                    <span className="bg-white/10 border border-white/10 text-white px-2 py-1 rounded-md">{m.version}</span>
                    <span className="opacity-50">•</span>
                    <span className="text-zinc-400">{m.architecture}</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-8 relative z-10">
              <div className="bg-black/40 rounded-2xl p-4 border border-white/5 shadow-inner">
                <div className="text-zinc-500 text-[10px] font-bold tracking-[0.2em] mb-2 flex items-center gap-2 uppercase">
                  <Target className="w-3.5 h-3.5 text-zinc-400" /> Validation Acc
                </div>
                <div className="text-3xl font-extrabold text-white">
                  {m.accuracy ? `${(m.accuracy * 100).toFixed(2)}%` : 'N/A'}
                </div>
              </div>
              <div className="bg-black/40 rounded-2xl p-4 border border-white/5 shadow-inner">
                <div className="text-zinc-500 text-[10px] font-bold tracking-[0.2em] mb-2 flex items-center gap-2 uppercase">
                  <Calendar className="w-3.5 h-3.5 text-zinc-400" /> Compiled On
                </div>
                <div className="text-lg font-bold text-white mt-1 truncate">
                  {new Date(m.created_at).toLocaleDateString()}
                </div>
              </div>
            </div>

            <div className="space-y-4 pt-6 border-t border-white/10 relative z-10">
              <div>
                <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-[0.2em] mb-3">Recognized Classes ({m.classes.length})</p>
                <div className="flex flex-wrap gap-2">
                  {m.classes.slice(0, 8).map((c: string, i: number) => (
                    <span key={i} className="text-xs bg-white/5 border border-white/10 text-zinc-300 px-3 py-1.5 rounded-lg font-medium tracking-wide">
                      {c}
                    </span>
                  ))}
                  {m.classes.length > 8 && (
                    <span className="text-xs text-zinc-500 bg-black/50 border border-white/5 px-3 py-1.5 rounded-lg font-bold">+{m.classes.length - 8}</span>
                  )}
                </div>
              </div>

              <div className="pt-6 flex justify-end gap-3 mt-4">
                {!m.is_active && (
                  <button 
                    onClick={() => activateModel(m.id)}
                    className="bg-white/5 border border-white/10 hover:border-white/30 hover:bg-white/10 text-white font-bold tracking-widest uppercase text-xs py-3 px-6 rounded-xl transition-all duration-300"
                  >
                    Engage Model
                  </button>
                )}
                {m.is_active && (
                  <div className="flex items-center text-xs font-bold tracking-[0.1em] uppercase text-white gap-3 bg-white/5 px-4 py-2 rounded-xl border border-white/10">
                    <span className="relative flex h-2.5 w-2.5">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-white"></span>
                    </span>
                    Primary Inference Core
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}

        {models.length === 0 && (
          <div className="col-span-full py-20 text-center border border-dashed border-white/10 rounded-3xl bg-white/[0.01]">
            <ShieldAlert className="w-16 h-16 text-zinc-700 mx-auto mb-6 opacity-50" />
            <h3 className="text-2xl font-bold text-zinc-400 mb-2 tracking-wide uppercase">No Models In Registry</h3>
            <p className="text-zinc-500 text-sm tracking-widest uppercase">Execute a training run to compile a model.</p>
          </div>
        )}
      </div>
    </div>
  );
}
