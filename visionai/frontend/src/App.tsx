import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, NavLink as RouterNavLink, useLocation, useNavigate } from 'react-router-dom';
import { Home, Database, BrainCircuit, Play, History, Settings, ScanEye, Terminal, Cpu, Users } from 'lucide-react';
import axios from 'axios';
import Dashboard from './pages/Dashboard';
import Datasets from './pages/Datasets';
import Training from './pages/Training';
import Predict from './pages/Predict';
import Models from './pages/Models';
import Registry from './pages/Registry';
import SettingsPage from './pages/Settings';
import { getApiUrl } from './config';

function AppContent() {
  const location = useLocation();
  const navigate = useNavigate();
  const [isMinimized, setIsMinimized] = useState(false);
  const [isClosed, setIsClosed] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const [sysInfo, setSysInfo] = useState<any>(null);
  const [projectMode, setProjectMode] = useState<'vision' | 'placement'>(() => {
    return (localStorage.getItem('visionai_project_mode') as 'vision' | 'placement') || 'vision';
  });

  const handleProjectModeChange = (mode: 'vision' | 'placement') => {
    setProjectMode(mode);
    localStorage.setItem('visionai_project_mode', mode);
    window.dispatchEvent(new Event('projectModeChanged'));
  };

  // Fetch system info for top menu bar
  useEffect(() => {
    const fetchSysInfo = async () => {
      try {
        const apiUrl = getApiUrl();
        const res = await axios.get(`${apiUrl}/system/info`);
        setSysInfo(res.data);
      } catch (e) {
        console.error("Failed to load system info", e);
      }
    };
    fetchSysInfo();
  }, []);

  // Map route to labels
  const getActiveTitle = () => {
    switch (location.pathname) {
      case '/': return 'Dashboard';
      case '/datasets': return 'Datasets';
      case '/training': return 'Train Model';
      case '/predict': return 'Inference & Prediction';
      case '/models': return 'Model Directory';
      case '/settings': return 'Preferences';
      default: return 'VisionAI Core';
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-gradient-to-tr from-zinc-950 via-slate-900 to-zinc-950 overflow-hidden relative select-none">
      {/* Animated Desktop Wallpaper Blobs */}
      <div className="absolute inset-0 z-0 opacity-20 pointer-events-none">
        <div className="absolute top-[-20%] left-[-20%] w-[50%] h-[50%] bg-blue-500/20 rounded-full mix-blend-screen filter blur-[120px] animate-[pulse-glow_8s_infinite]" />
        <div className="absolute bottom-[-20%] right-[-20%] w-[50%] h-[50%] bg-purple-500/20 rounded-full mix-blend-screen filter blur-[120px] animate-[pulse-glow_10s_infinite_2s]" />
      </div>



      {/* macOS Desktop Area */}
      <div className="flex-grow w-full relative flex items-center justify-center p-0 pb-14 md:p-4 md:pb-24 z-10">
        {isClosed ? (
          <div className="flex flex-col items-center gap-4 animate-mac-scale">
            <div className="w-20 h-20 rounded-3xl bg-zinc-900 border border-white/10 flex items-center justify-center shadow-2xl cursor-pointer hover:scale-105 active:scale-95 transition-all duration-300 animate-pulse" onClick={() => setIsClosed(false)}>
              <ScanEye className="w-10 h-10 text-white" />
            </div>
            <p className="text-zinc-500 text-xs font-bold uppercase tracking-[0.2em]">Launch Neural Core</p>
          </div>
        ) : isMinimized ? (
          <div className="text-center text-zinc-500 text-xs tracking-widest uppercase py-10 bg-zinc-950/40 backdrop-blur px-8 rounded-2xl border border-white/5 animate-mac-scale">
            Application Minimized to Dock
          </div>
        ) : (
          /* macOS Main Application Window */
          <div className={`w-full h-full md:w-[96%] md:h-[94%] md:max-w-6xl md:max-h-[78vh] md:rounded-3xl border-0 md:border md:border-white/10 flex flex-col md:flex-row overflow-hidden mac-glass z-20 ${isMaximized ? 'w-full h-full border-0 rounded-none' : ''} animate-mac-scale`}>
            {/* Sidebar */}
            <div className="hidden md:flex w-full md:w-56 border-b md:border-b-0 md:border-r border-white/10 bg-zinc-950/45 backdrop-blur-3xl flex-col flex-shrink-0">
              {/* Traffic Lights Controls Header */}
              <div className="h-12 flex items-center px-4 justify-between border-b border-white/5 flex-shrink-0">
                <div className="flex items-center gap-2 group">
                  <button 
                    className="w-3 h-3 rounded-full bg-[#ff5f56] border border-[#e0443e] flex items-center justify-center text-[7px] text-zinc-900 font-bold"
                    onClick={() => setIsClosed(true)}
                  >
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity">×</span>
                  </button>
                  <button 
                    className="w-3 h-3 rounded-full bg-[#ffbd2e] border border-[#dfa224] flex items-center justify-center text-[7px] text-zinc-900 font-bold"
                    onClick={() => setIsMinimized(true)}
                  >
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity">−</span>
                  </button>
                  <button 
                    className="w-3 h-3 rounded-full bg-[#27c93f] border border-[#1a9c2b] flex items-center justify-center text-[7px] text-zinc-900 font-bold"
                    onClick={() => setIsMaximized(!isMaximized)}
                  >
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity">⤢</span>
                  </button>
                </div>
                <div className="flex items-center gap-1.5 opacity-50">
                  <Terminal className="w-3.5 h-3.5" />
                  <span className="text-[10px] font-bold tracking-widest font-mono">CORE</span>
                </div>
              </div>

              {/* Sidebar Navigation */}
              <div className="flex-grow p-3 space-y-6 overflow-y-auto">
                <div className="pb-3 border-b border-white/5">
                  <h4 className="px-3 text-[9px] font-bold text-zinc-500 uppercase tracking-[0.2em] mb-2">Operational Project</h4>
                  <div className="px-3">
                    <select
                      value={projectMode}
                      onChange={(e) => handleProjectModeChange(e.target.value as 'vision' | 'placement')}
                      className="w-full bg-black/40 border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-white/20 transition-all font-bold tracking-wide uppercase cursor-pointer"
                    >
                      <option value="vision">🖼️ Vision AI</option>
                      <option value="placement">📊 Placement ANN</option>
                    </select>
                  </div>
                </div>

                <div>
                  <h4 className="px-3 text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em] mb-2.5">Neural Library</h4>
                  <nav className="space-y-1">
                    <NavLink to="/" icon={<Home className="w-4 h-4" />} label="Dashboard" />
                    <NavLink to="/datasets" icon={<Database className="w-4 h-4" />} label="Datasets" />
                    <NavLink to="/training" icon={<BrainCircuit className="w-4 h-4" />} label="Train Model" />
                  </nav>
                </div>

                <div>
                  <h4 className="px-3 text-[10px] font-bold text-zinc-500 uppercase tracking-[0.2em] mb-2.5">Inference</h4>
                  <nav className="space-y-1">
                    <NavLink to="/predict" icon={<Play className="w-4 h-4" />} label="Run Predict" />
                    <NavLink to="/models" icon={<History className="w-4 h-4" />} label="Models" />
                    {projectMode === 'placement' && (
                      <NavLink to="/registry" icon={<Users className="w-4 h-4" />} label="Student Registry" />
                    )}
                  </nav>
                </div>

                <div className="pt-2 border-t border-white/5">
                  <nav className="space-y-1">
                    <NavLink to="/settings" icon={<Settings className="w-4 h-4" />} label="Preferences" />
                  </nav>
                </div>
              </div>

              {/* System Footer info */}
              {sysInfo && (
                <div className="p-4 border-t border-white/5 bg-black/20 hidden md:block">
                  <div className="flex items-center gap-2 text-zinc-400 mb-1.5">
                    <Cpu className="w-3.5 h-3.5 text-zinc-400" />
                    <span className="text-[10px] font-bold tracking-wider font-mono">ENV STATUS</span>
                  </div>
                  <div className="text-[9px] text-zinc-500 space-y-1 font-mono uppercase">
                    <div>OS: {sysInfo.os_name} {sysInfo.os_release}</div>
                    <div>Cores: {sysInfo.cpu_count} Threaded</div>
                    <div>Torch: {sysInfo.pytorch_version}</div>
                  </div>
                </div>
              )}
            </div>

            {/* Canvas Area */}
            <div className="flex-1 flex flex-col overflow-hidden bg-[#18181f]/40 backdrop-blur-md">
              {/* macOS Translucent App Window Header Toolbar */}
              <div className="h-12 border-b border-white/10 flex items-center justify-between px-6 flex-shrink-0">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1">
                    <button 
                      className="w-6 h-6 rounded bg-white/5 hover:bg-white/10 border border-white/5 flex items-center justify-center text-xs text-zinc-400 hover:text-white mac-btn-spring"
                      onClick={() => navigate(-1)}
                    >
                      ‹
                    </button>
                    <button 
                      className="w-6 h-6 rounded bg-white/5 hover:bg-white/10 border border-white/5 flex items-center justify-center text-xs text-zinc-400 hover:text-white mac-btn-spring"
                      onClick={() => navigate(1)}
                    >
                      ›
                    </button>
                  </div>
                  <h2 className="text-sm font-bold text-white tracking-wide">{getActiveTitle()}</h2>
                </div>
                
                {/* Search / Filter bar (Desktop) & Project Mode Switcher (Mobile) */}
                <div className="flex items-center gap-2">
                  <select
                    value={projectMode}
                    onChange={(e) => handleProjectModeChange(e.target.value as 'vision' | 'placement')}
                    className="md:hidden bg-white/5 border border-white/10 rounded-lg px-2.5 py-1 text-xs text-white focus:outline-none focus:border-white/20 transition-all font-bold tracking-wide uppercase cursor-pointer"
                  >
                    <option value="vision">🖼️ Vision AI</option>
                    <option value="placement">📊 Placement ANN</option>
                  </select>

                  <div className="hidden md:flex w-48 lg:w-64 h-7 rounded-lg bg-black/30 border border-white/5 items-center px-3 gap-2">
                    <span className="text-zinc-500 text-[10px] font-bold font-mono">CMD + F</span>
                    <input 
                      type="text" 
                      placeholder="Search parameters..." 
                      className="bg-transparent text-xs w-full focus:outline-none text-zinc-300 placeholder-zinc-500"
                      disabled
                    />
                  </div>
                </div>
              </div>

              {/* Window Workspace Canvas (Content Router) */}
              <div className="flex-grow overflow-y-auto p-4 md:p-8">
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/datasets" element={<Datasets />} />
                  <Route path="/training" element={<Training />} />
                  <Route path="/predict" element={<Predict />} />
                  <Route path="/registry" element={<Registry />} />
                  <Route path="/models" element={<Models />} />
                  <Route path="/settings" element={<SettingsPage />} />
                </Routes>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Responsive Dock (Floating macOS Dock on Desktop, Tabbed bottom-bar on Mobile) */}
      <div className="fixed bottom-0 left-0 w-full h-14 md:absolute md:bottom-4 md:left-1/2 md:-translate-x-1/2 md:w-auto md:h-16 md:px-4 md:rounded-2xl mac-glass-bright flex items-center justify-around md:justify-center md:gap-2.5 pb-1 md:pb-2.5 z-40 shadow-2xl border-t border-white/5 md:border-t-0">
        <DockIcon to="/" active={location.pathname === '/'} icon={<Home className="w-6 h-6" />} tooltip="Dashboard" onClick={() => { setIsClosed(false); setIsMinimized(false); }} />
        <DockIcon to="/datasets" active={location.pathname === '/datasets'} icon={<Database className="w-6 h-6" />} tooltip="Datasets" onClick={() => { setIsClosed(false); setIsMinimized(false); }} />
        <DockIcon to="/training" active={location.pathname === '/training'} icon={<BrainCircuit className="w-6 h-6" />} tooltip="Training" onClick={() => { setIsClosed(false); setIsMinimized(false); }} />
        <DockIcon to="/predict" active={location.pathname === '/predict'} icon={<Play className="w-6 h-6" />} tooltip="Predict" onClick={() => { setIsClosed(false); setIsMinimized(false); }} />
        {projectMode === 'placement' && (
          <DockIcon to="/registry" active={location.pathname === '/registry'} icon={<Users className="w-6 h-6" />} tooltip="Registry" onClick={() => { setIsClosed(false); setIsMinimized(false); }} />
        )}
        <DockIcon to="/models" active={location.pathname === '/models'} icon={<History className="w-6 h-6" />} tooltip="Models" onClick={() => { setIsClosed(false); setIsMinimized(false); }} />
        <div className="hidden md:block w-[1px] h-10 bg-white/10 self-center mx-1" />
        <DockIcon to="/settings" active={location.pathname === '/settings'} icon={<Settings className="w-6 h-6" />} tooltip="Settings" onClick={() => { setIsClosed(false); setIsMinimized(false); }} />
      </div>
    </div>
  );
}

function NavLink({ to, icon, label }: { to: string, icon: React.ReactNode, label: string }) {
  return (
    <RouterNavLink 
      to={to} 
      className={({ isActive }) => 
        `flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-semibold tracking-wide transition-all duration-300 mac-btn-spring ` +
        (isActive 
          ? `bg-white/[0.08] border border-white/10 text-white shadow-sm` 
          : `text-zinc-400 hover:text-white hover:bg-white/[0.03] border border-transparent`)
      }
    >
      <div className="w-4 h-4 flex-shrink-0 flex items-center justify-center">{icon}</div>
      <span className="truncate">{label}</span>
    </RouterNavLink>
  );
}

function DockIcon({ to, active, icon, tooltip, onClick }: { to: string, active: boolean, icon: React.ReactNode, tooltip: string, onClick?: () => void }) {
  return (
    <RouterNavLink 
      to={to}
      onClick={onClick}
      className="mac-dock-item w-11 h-11 rounded-xl flex flex-col items-center justify-center bg-white/[0.03] hover:bg-white/[0.1] border border-white/5 hover:border-white/10 text-zinc-300 hover:text-white shadow-lg relative group active:scale-90"
    >
      {icon}
      
      {/* Active Dot Indicator */}
      {active && (
        <div className="w-1.5 h-1.5 rounded-full bg-white absolute -bottom-1" />
      )}
      
      {/* macOS Tooltip (Desktop Only) */}
      <div className="hidden md:block absolute -top-10 left-1/2 -translate-x-1/2 px-2.5 py-1 bg-zinc-900/90 border border-white/10 text-[10px] font-bold text-white rounded-lg opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-200 tracking-wider uppercase whitespace-nowrap shadow-xl">
        {tooltip}
      </div>
    </RouterNavLink>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  );
}

export default App;
