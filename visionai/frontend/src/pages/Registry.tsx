import { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  Search, Sliders, Users, CheckCircle, XCircle, Brain, Target, 
  ArrowUpRight, Award, FolderHeart, Star, Activity, Sparkles
} from 'lucide-react';
import { ResponsiveContainer, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Bar } from 'recharts';
import { getApiUrl } from '../config';

interface Student {
  name: string;
  cgpa: number;
  aptitude: number;
  comm: number;
  coding: number;
  internship: string;
  projects: number;
  placement: string;
}

export default function Registry() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [students, setStudents] = useState<Student[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);
  const [activeModel, setActiveModel] = useState<any>(null);
  
  // Real-time model evaluation state
  const [evaluating, setEvaluating] = useState(false);
  const [evalResult, setEvalResult] = useState<any>(null);

  const apiUrl = getApiUrl();

  useEffect(() => {
    // 1. Fetch Tabular Datasets
    axios.get(`${apiUrl}/datasets`).then(res => {
      const placementDatasets = res.data.filter((ds: any) => 
        ds.name.toLowerCase().includes('placement') || ds.name.toLowerCase().includes('recruitment')
      );
      setDatasets(placementDatasets);
      if (placementDatasets.length > 0) {
        setSelectedDatasetId(placementDatasets[0].id.toString());
      }
    }).catch(err => console.error(err));

    // 2. Fetch Active Tabular Model for real-time validation
    axios.get(`${apiUrl}/models`).then(res => {
      const activeTabular = res.data.find((m: any) => m.is_active && m.architecture.startsWith('ANN'));
      setActiveModel(activeTabular);
    }).catch(err => console.error(err));
  }, []);

  useEffect(() => {
    if (!selectedDatasetId) return;
    
    // Fetch students list and stats for selected dataset
    axios.get(`${apiUrl}/datasets/${selectedDatasetId}/students?search=${encodeURIComponent(searchQuery)}`)
      .then(res => {
        setStudents(res.data.students);
        setStats(res.data.stats);
      })
      .catch(err => console.error(err));
  }, [selectedDatasetId, searchQuery]);

  // Run model inference on the selected student
  const runModelEvaluation = async (student: Student) => {
    if (!activeModel) return;
    setEvaluating(true);
    setEvalResult(null);
    try {
      const res = await axios.post(`${apiUrl}/predict/tabular`, {
        cgpa: student.cgpa,
        aptitude_score: student.aptitude,
        communication_skills: student.comm,
        coding_skills: student.coding,
        internship: student.internship,
        projects_completed: student.projects
      });
      setEvalResult(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setEvaluating(false);
    }
  };

  const handleSelectStudent = (student: Student) => {
    setSelectedStudent(student);
    setEvalResult(null);
    if (activeModel) {
      runModelEvaluation(student);
    }
  };

  // Prepares data for Recharts comparison
  const getComparisonData = (student: Student) => {
    if (!stats) return [];
    
    // Normalize values to 0-100 scale for visual comparison
    // CGPA: score * 10
    // Aptitude: score
    // Coding: score * 20
    // Comm: score * 20
    // Projects: score * 20
    return [
      {
        metric: 'CGPA',
        'Student Score': student.cgpa * 10,
        'Placed Avg': stats.placed.cgpa * 10,
        'Unplaced Avg': stats.unplaced.cgpa * 10,
      },
      {
        metric: 'Aptitude',
        'Student Score': student.aptitude,
        'Placed Avg': stats.placed.aptitude,
        'Unplaced Avg': stats.unplaced.aptitude,
      },
      {
        metric: 'Coding',
        'Student Score': student.coding * 20,
        'Placed Avg': stats.placed.coding * 20,
        'Unplaced Avg': stats.unplaced.coding * 20,
      },
      {
        metric: 'Comm',
        'Student Score': student.comm * 20,
        'Placed Avg': stats.placed.comm * 20,
        'Unplaced Avg': stats.unplaced.comm * 20,
      },
      {
        metric: 'Projects',
        'Student Score': student.projects * 20,
        'Placed Avg': stats.placed.projects * 20,
        'Unplaced Avg': stats.unplaced.projects * 20,
      }
    ];
  };

  // Feedback mapping
  const generateRecommendations = (student: Student) => {
    const tips = [];
    if (student.cgpa < 7.5) {
      tips.push("CGPA is below the target average (7.5+). Focus on academic scores in remaining semesters to boost shortlist criteria.");
    }
    if (student.coding < 4) {
      tips.push("Coding capability score is moderate. Recommendation: Take up additional coding certification bootcamps or practice on LeetCode.");
    }
    if (student.projects < 2) {
      tips.push("Completed projects are low. Build at least 2 full-stack or data science portfolio projects to demonstrate practical coding skills.");
    }
    if (student.internship === 'No') {
      tips.push("Lacks internship experience. Target 4-8 week remote internships or project-based internships to bolster resume weight.");
    }
    if (student.comm < 4) {
      tips.push("Communication skill level is below placed student average. Participate in mock interviews and presentation workshops.");
    }
    if (tips.length === 0) {
      tips.push("Excellent profile! All parameters meet or exceed aggregate placement standards. Maintain current performance.");
    }
    return tips;
  };

  return (
    <div className="space-y-10 animate-in fade-in slide-in-from-bottom-8 duration-700 pb-16">
      <header className="relative">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white mb-2 uppercase">
          Student Registry<span className="text-zinc-500">.</span>
        </h1>
        <p className="text-zinc-400 text-lg tracking-wide max-w-xl font-light">
          Search student records, perform metric profile analyses, and validate active placement predictions.
        </p>
      </header>

      {/* Dataset Selector & Search */}
      <div className="glass-panel p-6 rounded-3xl border border-white/10 flex flex-col md:flex-row gap-4 items-center justify-between">
        <div className="flex items-center gap-3 w-full md:w-1/3">
          <Sliders className="w-5 h-5 text-zinc-500 shrink-0" />
          <select 
            value={selectedDatasetId} 
            onChange={e => { setSelectedDatasetId(e.target.value); setSelectedStudent(null); }}
            className="w-full bg-black/50 border border-white/10 rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-white/30 transition-colors cursor-pointer text-xs font-bold uppercase tracking-wider"
          >
            {datasets.map((ds: any) => (
              <option key={ds.id} value={ds.id}>{ds.name} ({ds.total_images} records)</option>
            ))}
          </select>
        </div>

        <div className="relative w-full md:w-1/2">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <input 
            type="text" 
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="Search student profile by name..." 
            className="w-full bg-black/50 border border-white/10 rounded-xl pl-12 pr-4 py-2.5 text-white focus:outline-none focus:border-white/30 transition-all placeholder:text-zinc-600 text-sm"
          />
        </div>
      </div>

      {/* Split Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Student list column */}
        <div className="lg:col-span-1 glass-panel rounded-3xl border border-white/10 p-6 max-h-[600px] overflow-y-auto space-y-4">
          <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-widest mb-2 flex items-center gap-2">
            <Users className="w-4 h-4" />
            Students List ({students.length})
          </h3>
          
          <div className="space-y-3">
            {students.map((st) => (
              <div 
                key={st.name} 
                onClick={() => handleSelectStudent(st)}
                className={`p-4 rounded-2xl border transition-all duration-300 cursor-pointer ${
                  selectedStudent?.name === st.name 
                    ? 'bg-white/10 border-white/30 shadow-lg' 
                    : 'bg-white/5 border-white/5 hover:border-white/20 hover:bg-white/[0.08]'
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <h4 className="font-extrabold text-white text-sm tracking-wide">{st.name}</h4>
                  <span className={`px-2 py-0.5 rounded text-[9px] font-extrabold uppercase tracking-widest border ${
                    st.placement === 'Yes' 
                      ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' 
                      : 'bg-rose-500/10 border-rose-500/20 text-rose-400'
                  }`}>
                    {st.placement === 'Yes' ? 'Placed' : 'Not Placed'}
                  </span>
                </div>
                
                <div className="flex items-center gap-4 text-xs text-zinc-500 font-medium">
                  <span>CGPA: <strong className="text-zinc-300 font-mono">{st.cgpa.toFixed(2)}</strong></span>
                  <span>Aptitude: <strong className="text-zinc-300 font-mono">{st.aptitude}</strong></span>
                  <span>Internship: <strong className="text-zinc-300">{st.internship}</strong></span>
                </div>
              </div>
            ))}

            {students.length === 0 && (
              <div className="text-center py-10 text-zinc-600 font-mono text-xs">
                No matching profiles found
              </div>
            )}
          </div>
        </div>

        {/* Selected Student profile column */}
        <div className="lg:col-span-2 space-y-6">
          {selectedStudent ? (
            <div className="glass-panel rounded-3xl border border-white/10 p-8 shadow-2xl space-y-8 animate-in slide-in-from-right-8 duration-500">
              
              {/* Header profile info */}
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 pb-6 border-b border-white/10">
                <div>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Student Performance Record</span>
                  <h2 className="text-3xl font-black text-white tracking-tight mt-1">{selectedStudent.name}</h2>
                </div>

                <div className="flex items-center gap-4">
                  <div className="flex flex-col items-end">
                    <span className="text-[9px] font-bold text-zinc-500 uppercase tracking-wider">HISTORIC STATUS</span>
                    <span className={`mt-1 flex items-center gap-1.5 px-3.5 py-1 rounded-full text-xs font-black uppercase tracking-wider border ${
                      selectedStudent.placement === 'Yes'
                        ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.1)]'
                        : 'bg-rose-500/10 border-rose-500/30 text-rose-400 shadow-[0_0_15px_rgba(239,68,68,0.1)]'
                    }`}>
                      {selectedStudent.placement === 'Yes' ? (
                        <><CheckCircle className="w-3.5 h-3.5" /> Placed</>
                      ) : (
                        <><XCircle className="w-3.5 h-3.5" /> Not Placed</>
                      )}
                    </span>
                  </div>

                  {activeModel && (
                    <div className="flex flex-col items-end border-l border-white/10 pl-4">
                      <span className="text-[9px] font-bold text-zinc-500 uppercase tracking-wider">ACTIVE ANN PREDICT</span>
                      {evaluating ? (
                        <span className="text-xs text-zinc-400 font-mono mt-1.5 animate-pulse">Running pass...</span>
                      ) : evalResult ? (
                        <span className={`mt-1 flex items-center gap-1.5 px-3.5 py-1 rounded-full text-xs font-black uppercase tracking-wider border ${
                          evalResult.prediction === 'Placed'
                            ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.2)]'
                            : 'bg-rose-500/20 border-rose-500/50 text-rose-400 shadow-[0_0_15px_rgba(239,68,68,0.2)]'
                        }`}>
                          <Brain className="w-3.5 h-3.5" />
                          {(evalResult.placement_probability * 100).toFixed(0)}% Likelihood
                        </span>
                      ) : null}
                    </div>
                  )}
                </div>
              </div>

              {/* Grid of details */}
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-6">
                <div className="bg-black/20 border border-white/5 p-4 rounded-2xl relative overflow-hidden">
                  <div className="absolute top-3 right-3 text-zinc-600"><Award className="w-4 h-4" /></div>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block">CGPA</span>
                  <span className="text-2xl font-black text-white font-mono mt-2 block">{selectedStudent.cgpa.toFixed(2)}</span>
                  <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden mt-3">
                    <div className="h-full bg-white rounded-full" style={{ width: `${(selectedStudent.cgpa / 10) * 100}%` }} />
                  </div>
                </div>

                <div className="bg-black/20 border border-white/5 p-4 rounded-2xl relative overflow-hidden">
                  <div className="absolute top-3 right-3 text-zinc-600"><Target className="w-4 h-4" /></div>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block">Aptitude Score</span>
                  <span className="text-2xl font-black text-white font-mono mt-2 block">{selectedStudent.aptitude} <span className="text-xs text-zinc-500">/ 100</span></span>
                  <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden mt-3">
                    <div className="h-full bg-white rounded-full" style={{ width: `${selectedStudent.aptitude}%` }} />
                  </div>
                </div>

                <div className="bg-black/20 border border-white/5 p-4 rounded-2xl relative overflow-hidden">
                  <div className="absolute top-3 right-3 text-zinc-600"><FolderHeart className="w-4 h-4" /></div>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block">Projects Completed</span>
                  <span className="text-2xl font-black text-white font-mono mt-2 block">{selectedStudent.projects}</span>
                  <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden mt-3">
                    <div className="h-full bg-white rounded-full" style={{ width: `${(selectedStudent.projects / 5) * 100}%` }} />
                  </div>
                </div>

                <div className="bg-black/20 border border-white/5 p-4 rounded-2xl relative overflow-hidden">
                  <div className="absolute top-3 right-3 text-zinc-600"><Star className="w-4 h-4" /></div>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block">Comm Skills</span>
                  <div className="flex gap-1 mt-3">
                    {Array.from({ length: 5 }).map((_, idx) => (
                      <Star 
                        key={idx} 
                        className={`w-4 h-4 ${idx < selectedStudent.comm ? 'text-amber-400 fill-amber-400' : 'text-zinc-700'}`} 
                      />
                    ))}
                  </div>
                </div>

                <div className="bg-black/20 border border-white/5 p-4 rounded-2xl relative overflow-hidden">
                  <div className="absolute top-3 right-3 text-zinc-600"><Star className="w-4 h-4" /></div>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block">Coding Skills</span>
                  <div className="flex gap-1 mt-3">
                    {Array.from({ length: 5 }).map((_, idx) => (
                      <Star 
                        key={idx} 
                        className={`w-4 h-4 ${idx < selectedStudent.coding ? 'text-amber-400 fill-amber-400' : 'text-zinc-700'}`} 
                      />
                    ))}
                  </div>
                </div>

                <div className="bg-black/20 border border-white/5 p-4 rounded-2xl relative overflow-hidden">
                  <div className="absolute top-3 right-3 text-zinc-600"><Activity className="w-4 h-4" /></div>
                  <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-wider block">Internship Done</span>
                  <span className="text-xl font-black text-white uppercase mt-2 block">{selectedStudent.internship}</span>
                </div>
              </div>

              {/* Chart Comparison section */}
              {stats && (
                <div className="space-y-4">
                  <h3 className="text-xs font-bold text-zinc-500 uppercase tracking-widest flex items-center gap-2">
                    <ArrowUpRight className="w-4 h-4 text-white" />
                    Population Comparison Benchmarking
                  </h3>
                  
                  <div className="h-64 w-full bg-black/40 border border-white/5 rounded-3xl p-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={getComparisonData(selectedStudent)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                        <XAxis dataKey="metric" stroke="#71717a" fontSize={11} />
                        <YAxis stroke="#71717a" fontSize={11} domain={[0, 100]} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: 'rgba(9, 9, 11, 0.9)', borderColor: 'rgba(255, 255, 255, 0.1)', borderRadius: '12px' }}
                          labelStyle={{ color: '#fff', fontWeight: 'bold' }}
                        />
                        <Legend wrapperStyle={{ fontSize: '11px' }} />
                        <Bar dataKey="Student Score" fill="#ffffff" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="Placed Avg" fill="#10b981" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="Unplaced Avg" fill="#ef4444" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Feedback and Recommendations */}
              <div className="bg-white/[0.02] border border-white/10 rounded-3xl p-6 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-32 h-32 bg-amber-400/5 rounded-full blur-2xl pointer-events-none" />
                <h3 className="text-xs font-bold text-zinc-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-amber-400 animate-pulse" />
                  Neural Performance Diagnostic & Action Plan
                </h3>
                
                <ul className="space-y-3">
                  {generateRecommendations(selectedStudent).map((rec, idx) => (
                    <li key={idx} className="text-zinc-400 text-xs font-light leading-relaxed flex items-start gap-2.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-amber-400 mt-1.5 shrink-0" />
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>

            </div>
          ) : (
            <div className="glass-panel rounded-3xl border border-white/10 min-h-[500px] flex flex-col items-center justify-center text-center p-8">
              <svg className="w-24 h-24 text-zinc-700 opacity-20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="0.5">
                <circle cx="12" cy="12" r="10" />
                <circle cx="12" cy="12" r="6" />
                <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
              </svg>
              <h3 className="text-lg font-extrabold uppercase tracking-widest text-zinc-400 mt-6">Select a Student</h3>
              <p className="text-zinc-600 text-xs tracking-wide max-w-xs font-light mt-2 leading-relaxed">
                Choose a student profile from the list to fetch performance telemetry, benchmark data, and run neural inference analysis.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
