import React, { useState, useRef, Suspense } from 'react';
import { 
  Upload, 
  Activity, 
  FileText, 
  Brain, 
  Download,
  SplitSquareHorizontal,
  Maximize2,
  Minimize2,
  Mic,
  MicOff,
  MessageSquare,
  AlertTriangle,
  Zap,
  Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Canvas } from '@react-three/fiber';
import { View, Preload } from '@react-three/drei';
import BrainModel from './BrainModel';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE || import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [file, setFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [scanStatus, setScanStatus] = useState("");
  const [result, setResult] = useState(null);
  
  const [isExporting, setIsExporting] = useState(false);
  const [pdfPreviewUrl, setPdfPreviewUrl] = useState(null);
  
  // HUD State
  const [activeMode, setActiveMode] = useState('gradcam');
  const [sliderValue, setSliderValue] = useState(1.0);
  const [zoomLevel, setZoomLevel] = useState(1.0);
  const constraintsRef = useRef(null);
  const [isSpatialExpanded, setIsSpatialExpanded] = useState(false);
  const [isDeconstructed, setIsDeconstructed] = useState(false);
  const [selectedHotspot, setSelectedHotspot] = useState(null);
  
  // Innovation Modules State
  const [isInsightOpen, setIsInsightOpen] = useState(false);
  const [isMicActive, setIsMicActive] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [messages, setMessages] = useState([]);

  // --- Sub-Components ---

  const RiskExplainerSidecar = () => {
    if (!result || !result.clinical_narrative || !result.risk_metrics) return null;
    return (
      <motion.div 
        className={`clinical-sidecar ${isInsightOpen ? 'expanded' : 'collapsed'}`}
        layout
      >
        <div className="sidecar-header" onClick={() => setIsInsightOpen(!isInsightOpen)}>
          <Info size={18} className="accent-cyan" />
          <span>CLINICAL RISK INSIGHT</span>
          {!isInsightOpen && <div className="pulse-dot" />}
        </div>
        
        <AnimatePresence>
          {isInsightOpen && (
            <motion.div 
              className="sidecar-body"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
            >
              <div className="risk-metric-block">
                 <small>NEXUS RISK INDEX</small>
                 <div className="risk-value-large">
                    {result.risk_metrics.risk_score}<span>/100</span>
                 </div>
              </div>
              <div className="clinical-text">
                {result.clinical_narrative}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    );
  };

  const TargetHUD = ({ location }) => {
    if (!location) return null;
    return (
      <motion.div 
        className="clinical-target-crosshair"
        initial={{ opacity: 0, scale: 2 }}
        animate={{ opacity: 1, scale: 1 }}
        style={{ 
          left: `${location.x * 100}%`,
          top: `${location.y * 100}%`
        }}
      >
        <div className="crosshair-ring" />
        <div className="crosshair-id">TRGT-01</div>
      </motion.div>
    );
  };

  // --- Core Handlers ---

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setResult(null);
      setZoomLevel(1.0);
    }
  };

  const analyzeMRI = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setScanStatus("INITIALIZING AI CORE...");
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error("Inference engine failure");
      const data = await response.json();
      
      setResult(data);
      setMessages([{ role: 'assistant', content: `Analysis complete. Primary finding: ${data.label}. I've generated a clinical risk profile. How can I assist?` }]);
      setActiveMode('gradcam');
      setSliderValue(1.0);
      setIsAnalyzing(false);
      setIsInsightOpen(true); // Auto-open the sidecar on completion
    } catch (error) {
      console.error(error);
      alert("AI Analysis Pipeline Failed.");
      setIsAnalyzing(false);
    }
  };

  const generateReport = async () => {
    if (!result) return;
    setIsExporting(true);
    try {
      const response = await fetch(`${API_BASE}/api/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(result),
      });
      const blob = await response.blob();
      setPdfPreviewUrl(window.URL.createObjectURL(blob));
    } catch (error) {
      alert("PDF Export Failed.");
    } finally {
      setIsExporting(false);
    }
  };

  const fetchScoreCAM = async () => {
    if (!file || !result) return;
    setScanStatus("GENERATING SCORE-CAM...");
    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const resp = await fetch(`${API_BASE}/api/scorecam`, { method: 'POST', body: formData });
      const data = await resp.json();
      setResult(prev => ({ ...prev, images: { ...prev.images, scorecam: data.heatmap } }));
      setActiveMode('scorecam');
    } catch (e) {
      alert("Score-CAM Engine Offline.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleModeChange = (mode) => {
    if (mode === 'scorecam' && !result?.images?.scorecam) {
      fetchScoreCAM();
      return;
    }
    setActiveMode(mode);
    setSliderValue(mode === 'split' ? 0.5 : 1.0);
  };

  const startVoiceCommands = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return alert("Browser does not support Speech API");
    const recognition = new SpeechRecognition();
    recognition.onstart = () => setIsMicActive(true);
    recognition.onend = () => setIsMicActive(false);
    recognition.onresult = (e) => {
      const cmd = e.results[e.results.length - 1][0].transcript.toLowerCase();
      if (cmd.includes('nexus') || cmd.includes('astra')) {
        if (cmd.includes('zoom in')) setZoomLevel(p => Math.min(p + 0.5, 5));
        if (cmd.includes('zoom out')) setZoomLevel(p => Math.max(p - 0.5, 1));
        if (cmd.includes('grad-cam')) handleModeChange('gradcam');
        if (cmd.includes('report')) generateReport();
      }
    };
    recognition.start();
  };

  // --- Render Helpers ---

  const spatialMapVariants = {
    docked: { bottom: "2rem", left: "2rem", width: "280px", height: "280px", scale: 1, zIndex: 10 },
    expanded: { top: "50%", left: "50%", width: "80vw", height: "80vh", x: "-50%", y: "-50%", zIndex: 100 }
  };

  return (
    <div className="viewport-root">
      
      {/* 1. CINEMATIC MRI VIEWPORT */}
      {result ? (
        <div className="mri-cinematic-display fullscreen" onWheel={(e) => setZoomLevel(p => Math.min(Math.max(1, p + e.deltaY * -0.001), 5))}>
          <motion.div animate={{ scale: zoomLevel }} style={{ width: '100%', height: '100%', position: 'absolute' }}>
            <img src={`data:image/png;base64,${result.images[activeMode === 'enhanced' ? 'enhanced' : 'original']}`} className="mri-layer" alt="Scan" />
            {activeMode === 'split' ? (
               <>
                  <img src={`data:image/png;base64,${result.images.heatmap}`} className="mri-layer" style={{ clipPath: `inset(0 0 0 ${sliderValue * 100}%)` }} alt="Split" />
                  <div className="split-wipe-handle" style={{ left: `${sliderValue * 100}%` }} />
               </>
            ) : (
               <img 
                 src={`data:image/png;base64,${activeMode === 'scorecam' ? result.images.scorecam : result.images.heatmap}`} 
                 className="mri-layer heatmap-layer" 
                 style={{ opacity: (activeMode === 'gradcam' || activeMode === 'scorecam') ? sliderValue : 0 }} 
                 alt="Heatmap" 
               />
            )}
            <TargetHUD location={selectedHotspot} />
          </motion.div>
        </div>
      ) : (
        <div className="zero-state-dropzone">
          <input type="file" id="mri-upload" onChange={handleFileUpload} hidden />
          <label htmlFor="mri-upload" className="dropzone-area">
            {file ? (
              <div className="dropzone-content">
                <FileText size={48} className="accent-cyan" />
                <h2>{file.name}</h2>
                <button className="workflow-btn glow float-btn" onClick={(e) => { e.preventDefault(); analyzeMRI(); }}>
                  {isAnalyzing ? <Activity size={18} className="spin" /> : <Brain size={18} />} INITIATE INFERENCE
                </button>
              </div>
            ) : (
              <div className="dropzone-content">
                <Brain size={64} className="accent-cyan" style={{ opacity: 0.8 }} />
                <h2>AWAITING INPUT</h2>
                <p>Click or drag MRI scan here</p>
              </div>
            )}
          </label>
        </div>
      )}

      {/* 2. HUD MODULES (Left Telemetry) */}
      <div className="hud-module top-left">
        <div className="brand-header minimal">
          <Brain size={24} className="accent-cyan" />
          <div className="brand-text"><h1>NEURAL NEXUS</h1><p>v1.1 Diagnostic HUD</p></div>
        </div>
        <AnimatePresence>
          {result && (
            <motion.div className="telemetry-minimal" initial={{ x: -20, opacity: 0 }} animate={{ x: 0, opacity: 1 }}>
              <div className="metric-row"><span className="metric-label">DIAGNOSIS</span><span className="metric-value">{result.label.toUpperCase()}</span></div>
              <div className="metric-row">
                <span className="metric-label">CONFIDENCE</span>
                <span className="metric-value accent-cyan">{(result.confidence * 100).toFixed(1)}% <small>±{(result.uncertainty * 100).toFixed(1)}%</small></span>
              </div>
              
              {/* Composite Dashboard [NEW] */}
              <div className="risk-dashboard-mini">
                <div className="risk-stats">
                   <div className="risk-stat-item"><span>Entropy</span><small>{result.risk_metrics.entropy}</small></div>
                   <div className="risk-stat-item"><span>Asymmetry</span><small>{result.risk_metrics.asymmetry}</small></div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* 3. COPILOT 3D (Bottom Left) */}
      <AnimatePresence>
        {(isAnalyzing || result) && (
          <motion.div className="spatial-canvas-wrapper" variants={spatialMapVariants} animate={isSpatialExpanded ? "expanded" : "docked"}>
            <div className="copilot-header">
               <Brain size={12} className="accent-cyan" /> <span>3D SPATIAL MAP</span>
               <button onClick={() => setIsSpatialExpanded(!isSpatialExpanded)}>{isSpatialExpanded ? <Minimize2 size={12}/> : <Maximize2 size={12}/>}</button>
            </div>
            <div className="canvas-container-placeholder">
              <Canvas camera={{ position: [0, 0, 5] }}>
                <Suspense fallback={null}>
                  <BrainModel diagnosis={result?.label} tumorLocation={result?.tumor_location} onHotspotClick={setSelectedHotspot} scale={isSpatialExpanded ? 2.5 : 1} />
                </Suspense>
              </Canvas>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 4. ACTIONS & MODES (Right & Center) */}
      <AnimatePresence>
        {result && (
          <>
            <div className="hud-module top-right">
              <div className="action-stack">
                <button className="hud-action-btn" onClick={generateReport} disabled={isExporting}><FileText size={16}/> REPORT</button>
                <button className={`hud-action-btn ${isMicActive ? 'pulse' : ''}`} onClick={startVoiceCommands}><Mic size={16}/></button>
              </div>
            </div>

            <div className="hud-module bottom-center cinematic-hud">
              <div className="hud-pill-menu">
                {['raw', 'enhanced', 'split', 'gradcam', 'scorecam'].map(m => (
                  <button key={m} className={activeMode === m ? 'active' : ''} onClick={() => handleModeChange(m)}>{m.toUpperCase()}</button>
                ))}
              </div>
              { (activeMode === 'gradcam' || activeMode === 'split') && (
                <div className="hud-adjuster">
                  <input type="range" className="styled-slider" min="0" max="1" step="0.01" value={sliderValue} onChange={(e) => setSliderValue(parseFloat(e.target.value))} />
                </div>
              )}
            </div>

            {/* UPSTREAM RECONCILIATION: SIDECAR + ORACLE */}
            <div className="hud-module bottom-right">
               <RiskExplainerSidecar />
            </div>

            <button className="oracle-trigger" onClick={() => setIsChatOpen(true)}>
              <MessageSquare size={24} />
            </button>

            {isChatOpen && (
              <motion.div className="oracle-container" initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }}>
                <div className="oracle-header"><span>ORACLE ENGINE</span><button onClick={() => setIsChatOpen(false)}>×</button></div>
                <div className="oracle-body">
                  <div className="narrative-box">
                    <h3>Clinical Impression (BioMistral-7B)</h3>
                    <div className="narrative-content">{result.clinical_narrative}</div>
                  </div>
                  {messages.map((m, i) => <div key={i} className={`message ${m.role}`}>{m.content}</div>)}
                </div>
              </motion.div>
            )}
          </>
        )}
      </AnimatePresence>

      {/* 5. PDF PREVIEW */}
      <AnimatePresence>
        {pdfPreviewUrl && (
          <div className="preview-overlay">
            <div className="preview-modal">
              <div className="preview-header"><span>REPORT PREVIEW</span><button onClick={() => setPdfPreviewUrl(null)}>CLOSE</button></div>
              <iframe src={pdfPreviewUrl} className="preview-iframe" title="PDF" />
            </div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
