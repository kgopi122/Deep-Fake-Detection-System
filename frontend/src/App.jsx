import React, { useState, useEffect } from 'react';
import {
  Shield, Upload, AlertOctagon, CheckCircle, Search,
  Cpu, Eye, BookOpen, Bot, Server, X, Play, Activity,
  Lock, Zap, Globe
} from 'lucide-react';
import LoginPage from './pages/LoginPage';

// --- API CONFIGURATION ---
const API_BASE_URL = `${import.meta.env.VITE_API_URL}/api/v1/analysis`;

// --- API Service ---
const apiService = {
  analyzeDeepfake: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch(`${API_BASE_URL}/deepfake`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error(`Server Error: ${response.statusText}`);
      return await response.json();
    } catch (error) {
      console.error("Deepfake Analysis API Error:", error);
      throw error;
    }
  },
  verifyNews: async (textOrUrl) => {
    try {
      const response = await fetch(`${API_BASE_URL}/news`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: textOrUrl }),
      });
      if (!response.ok) throw new Error(`Server Error: ${response.statusText}`);
      return await response.json();
    } catch (error) {
      console.error("News Verification API Error:", error);
      throw error;
    }
  }
};

// --- Sub-Components ---

const Navbar = ({ activeTab, setActiveTab, isAuthenticated, guestTrials }) => (
  <nav className="fixed top-0 left-0 right-0 z-50 glass-panel border-b-0 border-b-white/5 h-20">
    <div className="max-w-7xl mx-auto px-4 h-full flex items-center justify-between">
      <div
        className="flex items-center gap-3 cursor-pointer group"
        onClick={() => setActiveTab('home')}
      >
        <div className="relative">
          <Shield className="h-8 w-8 text-cyan-400 group-hover:text-cyan-300 transition-colors" />
          <div className="absolute inset-0 bg-cyan-500 blur-xl opacity-20 group-hover:opacity-40 transition-opacity"></div>
        </div>
        <span className="text-2xl font-bold tracking-tight text-white group-hover:text-cyan-50 transition-colors">
          Veri<span className="text-cyan-400">True</span>
        </span>
      </div>

      <div className="hidden md:flex items-center gap-1">
        {['Home', 'Deepfake  Detector', 'News  Verifier'].map((item) => {
          const id = item.toLowerCase().replace(/\s+/g, '');
          const isActive = activeTab === (id === 'home' ? 'home' : id);
          return (
            <button
              key={item}
              onClick={() => setActiveTab(id === 'home' ? 'home' : id)}
              className={`
                relative px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-300
                ${isActive
                  ? 'text-cyan-950 bg-cyan-400 shadow-[0_0_20px_rgba(34,211,238,0.4)]'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
                }
              `}
            >
              {item}
            </button>
          );
        })}
        {!isAuthenticated && (
          <div className="mr-2 px-3 py-1.5 rounded-full text-xs font-bold bg-cyan-950/50 border border-cyan-500/30 text-cyan-400 flex items-center shadow-[0_0_10px_rgba(34,211,238,0.1)]">
            Free Trials Remaining: {Math.max(0, 2 - guestTrials)}
          </div>
        )}
        {isAuthenticated ? (
          <button
            onClick={() => {
              localStorage.removeItem("veritrue_token");
              localStorage.removeItem("veritrue_user");
              window.location.reload();
            }}
            className="ml-4 px-5 py-2.5 rounded-full text-sm font-medium border border-red-500/50 text-red-400 hover:bg-red-500 hover:text-white transition-colors"
          >
            Logout
          </button>
        ) : (
          <button
            onClick={() => window.location.href = '/login'}
            className="ml-4 px-5 py-2.5 rounded-full text-sm font-medium border border-cyan-500/50 text-cyan-400 hover:bg-cyan-500 hover:text-white transition-colors"
          >
            Login
          </button>
        )}
      </div>
    </div>
  </nav>
);

const Hero = ({ setActiveTab, isAuthenticated }) => (
  <div className="relative min-h-screen flex items-center justify-center pt-20 overflow-hidden">
    {/* Decorational light blobs */}
    <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none"></div>
    <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-[120px] pointer-events-none"></div>

    <div className="relative z-10 text-center max-w-5xl mx-auto px-6">
      <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-slate-800/50 border border-slate-700/50 text-cyan-400 text-xs font-mono mb-8 backdrop-blur-md animate-in fade-in slide-in-from-bottom duration-700">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
        </span>
        System Online • v2.4.0
      </div>

      <h1 className="text-6xl md:text-8xl font-black text-white mb-8 tracking-tighter leading-tight animate-in zoom-in duration-700 delay-100">
        Authenticity <br />
        <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 text-glow">
          Redefined
        </span>
      </h1>

      <p className="text-xl text-slate-400 mb-12 max-w-2xl mx-auto leading-relaxed animate-in slide-in-from-bottom duration-700 delay-200">
        Advanced forensic analysis for the digital age. Detect AI manipulation in images, video, and text with enterprise-grade precision.
      </p>

      <div className="flex flex-col sm:flex-row gap-6 justify-center animate-in slide-in-from-bottom duration-700 delay-300">
        <button
          onClick={() => setActiveTab('deepfakedetector')}
          className="group relative px-8 py-4 bg-cyan-500 hover:bg-cyan-400 text-slate-900 rounded-xl font-bold transition-all hover:scale-105 hover:shadow-[0_0_40px_rgba(34,211,238,0.4)]"
        >
          <div className="flex items-center gap-3">
            <ScanIcon className="group-hover:rotate-180 transition-transform duration-500" />
            <span>Analyze Media</span>
          </div>
        </button>

        <button
          onClick={() => setActiveTab('newsverifier')}
          className="px-8 py-4 glass-panel hover:bg-white/10 text-white rounded-xl font-bold transition-all hover:scale-105 flex items-center gap-3"
        >
          <Search size={20} />
          <span>Verify Source</span>
        </button>
      </div>

      {!isAuthenticated && (
        <div className="mt-8 animate-in slide-in-from-bottom duration-700 delay-400">
          <p className="text-slate-400 text-sm">
            <a href="/login" className="text-cyan-400 hover:text-cyan-300 underline underline-offset-4 transition-colors">Login</a> for unlimited uploads
          </p>
        </div>
      )}
    </div>
  </div>
);

const ScanIcon = ({ className }) => (
  <svg className={`w-5 h-5 ${className}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 7V5a2 2 0 0 1 2-2h2" />
    <path d="M17 3h2a2 2 0 0 1 2 2v2" />
    <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
    <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
  </svg>
)

const FileUploadCard = ({ onAnalyze, setSharedPreview }) => {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault(); e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]);
  };

  const processFile = (selectedFile) => {
    if (selectedFile && (selectedFile.type.startsWith('image/') || selectedFile.type.startsWith('video/'))) {
      setFile(selectedFile);
      const objectUrl = URL.createObjectURL(selectedFile);
      setPreview(objectUrl);
      setSharedPreview(objectUrl);
    } else {
      alert("Please upload a valid Image or Video file.");
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto animate-in zoom-in duration-500">
      <div
        className={`relative group glass-panel rounded-3xl p-12 transition-all duration-500 
          ${dragActive ? 'border-cyan-500/50 bg-cyan-900/10 scale-[1.02]' : ''}
        `}
        onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
      >
        <div className="flex flex-col items-center justify-center text-center">
          {file ? (
            <div className="w-full">
              <div className="relative w-full h-80 mb-8 rounded-2xl overflow-hidden bg-black/50 border border-white/10 group-hover:border-cyan-500/30 transition-colors">
                {/* Scanner Overlay */}
                <div className="absolute inset-x-0 h-[2px] bg-cyan-400 shadow-[0_0_20px_rgba(34,211,238,1)] z-10 animate-scan pointer-events-none"></div>

                {file.type.startsWith('image/') ? (
                  <img src={preview} alt="Preview" className="w-full h-full object-contain" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center flex-col text-slate-500">
                    <div className="w-20 h-20 rounded-full bg-slate-800 flex items-center justify-center mb-4">
                      <Play size={40} className="text-cyan-400 ml-1" />
                    </div>
                    <span className="font-mono text-sm uppercase tracking-widest">Video Analysis Ready</span>
                  </div>
                )}
                <button
                  onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null); setSharedPreview(null); }}
                  className="absolute top-4 right-4 p-2 bg-black/60 backdrop-blur text-white/70 hover:text-white rounded-full hover:bg-red-500/80 transition-all"
                >
                  <X size={20} />
                </button>
              </div>

              <div className="flex items-center justify-between mb-8 px-2">
                <div className="text-left">
                  <h3 className="text-lg font-bold text-white truncate max-w-[250px]">{file.name}</h3>
                  <p className="text-slate-400 text-xs font-mono mt-1">{(file.size / 1024 / 1024).toFixed(2)} MB • {file.type.toUpperCase()}</p>
                </div>
                <div className="flex items-center gap-2 text-cyan-400 text-sm bg-cyan-950/30 px-4 py-1.5 rounded-full border border-cyan-500/20 shadow-[0_0_15px_rgba(34,211,238,0.1)]">
                  <CheckCircle size={14} /><span>READY TO SCAN</span>
                </div>
              </div>

              <button
                onClick={() => onAnalyze(file)}
                className="w-full py-5 rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-600 text-white text-lg font-bold tracking-wide shadow-[0_0_30px_rgba(34,211,238,0.3)] hover:shadow-cyan-500/50 hover:scale-[1.01] active:scale-[0.99] transition-all"
              >
                INITIALIZE FORENSIC SCAN
              </button>
            </div>
          ) : (
            <>
              <div className="w-24 h-24 rounded-full bg-slate-800/50 flex items-center justify-center mb-8 group-hover:bg-cyan-900/20 group-hover:scale-110 transition-all duration-500">
                <Upload className="w-10 h-10 text-slate-400 group-hover:text-cyan-400 transition-colors" />
              </div>
              <h3 className="text-3xl font-bold text-white mb-4">Drop Media Evidence</h3>
              <p className="text-slate-400 mb-10 max-w-md mx-auto leading-relaxed">
                Upload images or video for deep neural network analysis.
                <br /><span className="text-xs opacity-50">Supported: JPG, PNG, MP4, WEBM</span>
              </p>
              <label className="cursor-pointer px-8 py-4 rounded-xl border border-slate-600 hover:border-cyan-500 hover:text-white text-slate-300 font-bold transition-all hover:bg-white/5">
                <span>BROWSE FILES</span>
                <input type="file" className="hidden" accept="image/*,video/*" onChange={(e) => e.target.files[0] && processFile(e.target.files[0])} />
              </label>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

const AnalysisLoader = ({ steps }) => {
  const [currentStep, setCurrentStep] = useState(0);
  useEffect(() => {
    if (currentStep < steps.length) {
      const timeout = setTimeout(() => setCurrentStep(c => c + 1), 1200 / steps.length);
      return () => clearTimeout(timeout);
    }
  }, [currentStep, steps.length]);

  return (
    <div className="w-full max-w-lg mx-auto py-20 animate-in fade-in">
      <div className="glass-panel p-8 rounded-2xl">
        <div className="flex items-center justify-center mb-8">
          <Cpu className="w-12 h-12 text-cyan-400 animate-spin-slow" />
        </div>
        <div className="mb-8 relative">
          <div className="flex justify-between text-[10px] text-cyan-400/70 font-mono mb-2 tracking-widest uppercase">
            <span>Progress</span><span>{Math.min(100, Math.round((currentStep / steps.length) * 100))}%</span>
          </div>
          <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden border border-white/5">
            <div className="h-full bg-cyan-400 relative transition-all duration-300 ease-out" style={{ width: `${(currentStep / steps.length) * 100}%` }}>
              <div className="absolute inset-0 bg-white/40 animate-shimmer"></div>
            </div>
          </div>
        </div>
        <div className="space-y-4">
          {steps.map((step, idx) => (
            <div key={idx} className={`flex items-center gap-4 transition-all duration-500 ${idx > currentStep ? 'opacity-30' : 'opacity-100'}`}>
              <div className={`w-2 h-2 rounded-full ${idx === currentStep ? 'bg-cyan-400 box-glow' : idx < currentStep ? 'bg-green-500' : 'bg-slate-700'}`}></div>
              <span className={`font-mono text-xs ${idx === currentStep ? 'text-cyan-400' : 'text-slate-400'}`}>{step}...</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const ResultGauge = ({ score, label, color }) => {
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-32 h-32">
        <svg className="w-full h-full transform -rotate-90 filter drop-shadow-[0_0_8px_rgba(0,0,0,0.5)]">
          <circle cx="64" cy="64" r={radius} stroke="#1e293b" strokeWidth="8" fill="transparent" />
          <circle cx="64" cy="64" r={radius} stroke="currentColor" strokeWidth="8" fill="transparent" strokeDasharray={circumference} strokeDashoffset={offset} strokeLinecap="round" className={`${color} transition-all duration-1000 ease-out`} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center flex-col">
          <span className="text-3xl font-bold text-white">{score.toFixed(0)}<span className="text-sm text-slate-500">%</span></span>
        </div>
      </div>
      <span className="mt-4 text-slate-400 font-bold uppercase tracking-widest text-[10px]">{label}</span>
    </div>
  );
};

const ResultsPanel = ({ data, onReset, previewUrl }) => {
  const deepfakeVerdict = data.type === 'deepfake'
    ? (data.evidence?.find((e) => e?.title === 'Executive Verdict')?.verdict || (data.fake ? 'DEEPFAKE' : 'REAL'))
    : null;

  const isSuspiciousDeepfake = data.type === 'deepfake' && deepfakeVerdict === 'SUSPICIOUS';
  const isDeepfake = data.type === 'deepfake' && (data.fake || deepfakeVerdict === 'DEEPFAKE');
  const isSafe = data.type === 'deepfake' ? (!isDeepfake && !isSuspiciousDeepfake) : data.credibility > 70;
  const [visualMode, setVisualMode] = useState('original');
  const accentColor = isDeepfake || !isSafe ? 'red' : 'green';

  return (
    <div className="max-w-7xl mx-auto animate-in fade-in slide-in-from-bottom duration-700">
      {/* Header Stat Card */}
      <div className={`glass-panel p-1 rounded-3xl mb-8 relative overflow-hidden`}>
        <div className={`absolute top-0 left-0 w-full h-1 ${isDeepfake || !isSafe ? 'bg-red-500' : 'bg-green-500'} shadow-[0_0_20px_currentColor]`}></div>
        <div className="bg-slate-950/80 rounded-[22px] p-8 md:p-10">
          <div className="flex flex-col md:flex-row items-center justify-between gap-8">
            <div className="flex items-center gap-8">
              <div className={`w-24 h-24 rounded-2xl flex items-center justify-center shadow-[0_0_30px_rgba(0,0,0,0.3)] ${isDeepfake || !isSafe ? 'bg-red-500 text-white' : 'bg-green-500 text-white'}`}>
                {isDeepfake || !isSafe ? <AlertOctagon size={48} /> : <Shield size={48} />}
              </div>
              <div>
                <div className="flex items-center gap-3 mb-2">
                  <div className={`h-2 w-2 rounded-full ${isDeepfake ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`}></div>
                  <span className="text-xs font-mono text-slate-400 uppercase tracking-widest">Analysis Complete</span>
                </div>
                <h2 className={`text-4xl md:text-5xl font-black tracking-tight mb-2 ${isDeepfake || !isSafe ? 'text-red-400' : 'text-green-400'}`}>
                  {data.type === 'deepfake'
                    ? (isDeepfake ? 'FAKE DETECTED' : (isSuspiciousDeepfake ? 'SUSPICIOUS MEDIA' : 'AUTHENTIC MEDIA'))
                    : (isSafe ? 'VERIFIED SOURCE' : 'SUSPICIOUS')}
                </h2>
                <p className="text-slate-400 text-lg">
                  Confidence: <span className="text-white font-bold">{data.type === 'deepfake' ? (data.confidence?.toFixed(1) || '0.0') : (data.credibility?.toFixed(1) || '0.0')}%</span>
                </p>
              </div>
            </div>

            <div className="bg-white/5 rounded-xl p-4 border border-white/5 backdrop-blur-sm">
              <div className="text-xs text-slate-500 uppercase tracking-widest mb-1">Scan ID</div>
              <div className="font-mono text-cyan-400 text-xl tracking-widest">{data.id ? data.id.toString().slice(-8) : "XJ9-001"}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-12">
        <div className="lg:col-span-8 space-y-8">
          {data.type === 'deepfake' ? (
            <div className="glass-panel p-1 rounded-2xl">
              <div className="bg-slate-950/80 rounded-xl overflow-hidden">
                <div className="p-4 border-b border-white/5 flex justify-between items-center bg-white/5">
                  <h3 className="font-bold text-white flex items-center gap-2 text-sm uppercase tracking-wider"><Eye size={16} className="text-cyan-400" />Forensic Vision</h3>
                  <div className="flex bg-black/50 rounded-lg p-1 gap-1">
                    {['original', 'heatmap', 'ela'].map((mode) => (
                      <button
                        key={mode}
                        onClick={() => setVisualMode(mode)}
                        className={`px-4 py-1.5 text-[10px] font-bold uppercase rounded-md transition-all 
                        ${visualMode === mode ? 'bg-cyan-600 text-white shadow-lg' : 'text-slate-500 hover:text-white'}`}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                </div>
                <div className={`relative aspect-video bg-black flex items-center justify-center overflow-hidden border-t border-white/5`}>
                  {/* Grid Overlay */}
                  <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none"></div>

                  {previewUrl && (
                    <img
                      src={previewUrl}
                      alt="Analysis Target"
                      className={`max-h-full max-w-full object-contain transition-all duration-300 
                        ${visualMode === 'ela' ? 'contrast-[150%] brightness-75 invert-[.1] sepia-[1] saturate-[500%] hue-rotate-[180deg]' : ''}
                      `}
                    />
                  )}
                  {visualMode === 'heatmap' && (
                    <div className="absolute inset-0 bg-gradient-to-tr from-blue-500/0 via-red-500/30 to-yellow-500/20 mix-blend-overlay animate-pulse"></div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="glass-panel p-8 rounded-2xl">
              <h3 className="font-bold text-white flex items-center gap-2 mb-8 uppercase tracking-widest text-sm"><BookOpen size={16} className="text-cyan-400" />Analysis Log</h3>
              <div className="space-y-4">
                {data.evidence && data.evidence.map((item, idx) => (
                  <div key={idx} className="bg-white/5 p-6 rounded-xl border border-white/5 hover:border-cyan-500/30 transition-all group">
                    <div className="flex justify-between items-start mb-3">
                      <span className={`text-[10px] uppercase font-bold px-3 py-1 rounded-full border ${item.verdict === 'False' ? 'border-red-500/30 text-red-400 bg-red-500/10' : 'border-green-500/30 text-green-400 bg-green-500/10'}`}>
                        {item.verdict}
                      </span>
                    </div>
                    <h4 className="text-white font-bold mb-2 group-hover:text-cyan-400 transition-colors">{item.title}</h4>
                    <p className="text-slate-400 text-sm leading-relaxed">{item.snippet}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Metrics */}
          {data.details && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="glass-panel p-6 rounded-2xl"><ResultGauge score={data.details.faceInconsistencies} label="Face Topology" color="text-orange-500" /></div>
              <div className="glass-panel p-6 rounded-2xl"><ResultGauge score={data.details.compressionArtifacts} label="Compression Noise" color="text-purple-500" /></div>
              <div className="glass-panel p-6 rounded-2xl"><ResultGauge score={data.details.lightingAnomalies} label="Light Vector Delta" color="text-cyan-500" /></div>
            </div>
          )}
        </div>

        <div className="lg:col-span-4 space-y-6">
          {/* Attribution Panel */}
          <div className="glass-panel p-8 rounded-2xl bg-gradient-to-b from-slate-900/60 to-slate-900/90">
            <h3 className="text-white text-xs font-bold mb-6 flex items-center gap-2 uppercase tracking-widest"><Bot size={14} className="text-cyan-400" />Origin Trace</h3>

            {data.type === 'deepfake' && data.attribution ? (
              <div className="space-y-8">
                <div className="relative p-6 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="absolute -top-3 left-4 px-2 bg-slate-900 text-xs text-slate-500">GENERATOR</div>
                  <div className="text-xl font-bold text-white flex items-center gap-3">
                    {data.attribution.generator}
                    <Zap size={16} className="text-yellow-400 ml-auto" />
                  </div>
                </div>

                <div>
                  <div className="space-y-3">
                    {data.attribution.methodology?.map((m, i) => (
                      <div key={i} className="flex items-center gap-3 text-sm text-slate-300">
                        <div className="w-1.5 h-1.5 rounded-full bg-cyan-500"></div>
                        {m}
                      </div>
                    ))}
                  </div>
                </div>

                {data.attribution.verifiedBy && (
                  <div>
                    <div className="text-xs text-slate-500 uppercase tracking-widest mb-4">Verification Nodes</div>
                    <div className="flex flex-wrap gap-2">
                      {data.attribution.verifiedBy.map((v, i) => (
                        <span key={i} className="px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs text-cyan-400 font-mono">
                          {v}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-slate-500 text-sm italic">Insufficient data for attribution model.</div>
            )}
          </div>
        </div>
      </div>

      <div className="text-center">
        <button
          onClick={onReset}
          className="group px-8 py-3 rounded-full bg-white/5 hover:bg-white/10 border border-white/10 text-white text-sm font-bold tracking-wide transition-all hover:scale-105 flex items-center gap-2 mx-auto"
        >
          <X size={16} className="text-slate-400 group-hover:text-white transition-colors" />
          START NEW INVESTIGATION
        </button>
      </div>
    </div>
  );
};

export default function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem("veritrue_token"));
  const [guestTrials, setGuestTrials] = useState(parseInt(localStorage.getItem('veritrue_guest_trials') || '0', 10));
  const [activeTab, setActiveTab] = useState('home');
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [urlInput, setUrlInput] = useState('');
  const [textInput, setTextInput] = useState('');
  const [sharedPreview, setSharedPreview] = useState(null);

  const handleAnalyze = async (data, type) => {
    if (!isAuthenticated) {
      if (guestTrials >= 2) {
        alert("Free trials exhausted! Redirecting to login.");
        window.location.href = '/login';
        return;
      }
    }

    if (type === 'news' && !urlInput && !textInput) {
      alert("Please enter a URL or Text to verify.");
      return;
    }
    setAnalyzing(true);
    setResult(null);
    try {
      let res;
      if (type === 'deepfake') {
        res = await apiService.analyzeDeepfake(data);
      } else {
        res = await apiService.verifyNews(urlInput || textInput);
      }
      setResult(res);
      
      if (!isAuthenticated) {
        const newTrials = guestTrials + 1;
        setGuestTrials(newTrials);
        localStorage.setItem('veritrue_guest_trials', newTrials.toString());
      }
    } catch (error) {
      console.error(error);
      alert("Backend Connection Failed. Ensure VeriTrue Server is running on Port 8000.");
    } finally {
      setAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setResult(null);
    setAnalyzing(false);
    setUrlInput('');
    setTextInput('');
    setSharedPreview(null);
  };

  if (window.location.pathname === '/login') {
    return <LoginPage />;
  }

  return (
    <div className="min-h-screen pt-20 pb-12">
      <Navbar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        isAuthenticated={isAuthenticated}
        guestTrials={guestTrials}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {activeTab === 'home' && <Hero setActiveTab={setActiveTab} isAuthenticated={isAuthenticated} />}

        {activeTab === 'deepfakedetector' && (
          <div className="max-w-6xl mx-auto py-12">
            {!analyzing && !result && <FileUploadCard onAnalyze={(f) => handleAnalyze(f, 'deepfake')} setSharedPreview={setSharedPreview} />}
            {analyzing && <AnalysisLoader steps={['UPLOADING TO NEURAL NET', 'PERFORMING BYTE-LEVEL FORENSICS', 'ANALYZING COMPRESSION ARTIFACTS', 'GENERATING FINAL REPORT']} />}
            {result && <ResultsPanel data={result} onReset={resetAnalysis} previewUrl={sharedPreview} />}
          </div>
        )}

        {activeTab === 'newsverifier' && (
          <div className="max-w-5xl mx-auto py-12">
            {!analyzing && !result && (
              <div className="glass-panel p-12 rounded-3xl max-w-3xl mx-auto text-center">
                <div className="w-20 h-20 rounded-full bg-slate-800/50 flex items-center justify-center mx-auto mb-6">
                  <Search className="w-10 h-10 text-cyan-400" />
                </div>
                <h2 className="text-3xl font-bold text-white mb-8">Verify Information Source</h2>
                <div className="relative mb-6">
                  <input
                    type="text"
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    placeholder="Paste article URL for verification..."
                    className="block w-full p-5 bg-slate-900/80 border border-slate-700 rounded-2xl text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all text-lg"
                  />
                </div>
                <button onClick={() => handleAnalyze({}, 'news')} className="w-full py-5 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-2xl shadow-lg shadow-cyan-900/20 hover:shadow-cyan-500/30 transition-all text-lg tracking-wide">
                  VERIFY AUTHENTICITY
                </button>
              </div>
            )}
            {analyzing && <AnalysisLoader steps={['CONNECTING TO KNOWLEDGE BASE', 'CROSS-REFERENCING SOURCES', 'CALCULATING CREDIBILITY SCORE']} />}
            {result && <ResultsPanel data={result} onReset={resetAnalysis} />}
          </div>
        )}
      </main>
    </div>
  );
}