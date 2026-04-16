import React, { useState, useRef, Suspense } from 'react';
import { 
  Upload, 
  Activity, 
  FileText, 
  Brain, 
  Download,
  SplitSquareHorizontal,
  Maximize2,
  Minimize2
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
  
  // Interactive Mechanics State
  const [activeMode, setActiveMode] = useState('gradcam');
  const [sliderValue, setSliderValue] = useState(1.0); // Acts as opacity OR wipe percentage
  const [zoomLevel, setZoomLevel] = useState(1.0);     // Advanced Interaction: Zoom target
  const constraintsRef = useRef(null);                 // Advanced Interaction: Drag bounds
  const [isSpatialExpanded, setIsSpatialExpanded] = useState(false);
  const [isDeconstructed, setIsDeconstructed] = useState(false);
  const [selectedHotspot, setSelectedHotspot] = useState(null);
  const [isInsightOpen, setIsInsightOpen] = useState(false);

  const RiskExplainerPanel = () => {
    if (!result || !result.clinical_narrative || !result.risk_metrics) return null;
    
    return (
      <motion.div 
        className={`nexus-oracle-sidecar ${isInsightOpen ? 'expanded' : 'collapsed'}`}
        layout
      >
        <div className="oracle-header" onClick={() => setIsInsightOpen(!isInsightOpen)}>
          <Brain size={18} className="accent-cyan" />
          <span>AI INSIGHT & RISK</span>
          {!isInsightOpen && <div className="unread-dot" />}
        </div>
        
        <AnimatePresence>
          {isInsightOpen && (
            <motion.div 
              className="oracle-body risk-panel-body"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 400, opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              style={{ overflowY: 'auto', padding: '1rem', whiteSpace: 'pre-wrap', lineHeight: '1.5', fontSize: '0.85rem' }}
            >
              <div style={{ marginBottom: '1rem', padding: '1rem', background: 'rgba(0,0,0,0.5)', borderRadius: '8px', borderLeft: '4px solid var(--accent-cyan)' }}>
                 <h3 style={{ margin: 0, color: 'var(--text-secondary)', fontSize: '0.7rem', letterSpacing: '1px' }}>CLINICAL RISK SCORE</h3>
                 <div style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'var(--accent-cyan)' }}>
                    {result.risk_metrics.risk_score}<span style={{ fontSize: '1rem', color: 'var(--text-secondary)' }}>/100</span>
                 </div>
              </div>
              <div style={{ color: 'var(--text-primary)' }}>
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

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setResult(null);
      setZoomLevel(1.0); // reset zoom on new upload
    }
  };

  const analyzeMRI = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setScanStatus("INITIALIZING NEURAL CORE...");
    const formData = new FormData();
    formData.append('file', file);

    try {
      setTimeout(() => setScanStatus("EXTRACTING FEATURES VIA RESNET-50..."), 800);
      setTimeout(() => setScanStatus("GENERATING GRAD-CAM HEATMAP..."), 1600);

      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error("Analysis engine failed");

      const data = await response.json();
      
      setTimeout(() => {
        setResult(data);
        setActiveMode('gradcam');
        setSliderValue(1.0);
        setZoomLevel(1.0);
        setIsAnalyzing(false);
        
        if (data.clinical_narrative) {
          setIsInsightOpen(true); // Auto-open risk panel to show the impression
        }
      }, 2400); 
    } catch (error) {
      console.error(error);
      alert("AI Core Analysis Failed. Please check backend status.");
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

      if (!response.ok) throw new Error("PDF generation failed");

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      setPdfPreviewUrl(url);
    } catch (error) {
      console.error(error);
      alert("PDF Generation failed. Check backend logs.");
    } finally {
      setIsExporting(false);
    }
  };

  const downloadPDF = () => {
    if (!pdfPreviewUrl) return;
    const a = document.createElement('a');
    a.href = pdfPreviewUrl;
    a.download = `Neural_Nexus_Report.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  const closePreview = () => {
    setPdfPreviewUrl(null);
  };

  // Wheel zoom handler
  const handleWheel = (e) => {
    if (!result) return;
    // Map scroll wheel to zoom (0.001 dampening for smooth scrolling)
    const zoomDelta = e.deltaY * -0.001;
    // Clamp zoom strictly between 1x and 5x
    setZoomLevel(prev => Math.min(Math.max(1.0, prev + zoomDelta), 5.0));
  };

  // When changing modes, if moving TO split, set default slider to 50%.
  const handleModeChange = (mode) => {
    setActiveMode(mode);
    if (mode === 'split') {
      setSliderValue(0.5); // Center the wipe separator
    } else if (mode === 'gradcam') {
      setSliderValue(1.0); // Full opacity
    }
  };
  const spatialMapVariants = {
    docked: {
      bottom: "2rem",
      left: "2rem",
      top: "auto",
      right: "auto",
      width: "280px",
      height: "280px",
      x: "0%",
      y: "0%",
      zIndex: 50,
      opacity: 1,
      scale: 1,
    },
    global: {
      top: "50%",
      left: "50%",
      bottom: "auto",
      right: "auto",
      width: "100vw",
      height: "100vh",
      x: "-50%",
      y: "-50%",
      zIndex: 100,
      opacity: 1,
      scale: 1,
    },
    expanded: {
      top: "50%",
      left: "50%",
      bottom: "auto",
      right: "auto",
      width: "80vw",
      height: "80vh",
      x: "-50%",
      y: "-50%",
      zIndex: 100,
      opacity: 1,
      scale: 1,
    }
  };

  return (
    <div className="viewport-root">
      
      {/* BACKGROUND / MAIN CONTENT MANAGER */}
      {result ? (
        <div 
          className="mri-cinematic-display mri-interactive-viewport fullscreen"
          ref={constraintsRef}
          onWheel={handleWheel}
        >
          <motion.div
            className="mri-layer-group"
            drag={zoomLevel > 1.0} // Only allow dragging if zoomed in
            dragConstraints={constraintsRef}
            animate={{ scale: zoomLevel }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
            style={{ width: '100%', height: '100%', position: 'absolute' }}
          >
            {/* Base Layer */}
            <img 
              src={`data:image/png;base64,${result.images[activeMode === 'enhanced' ? 'enhanced' : 'original']}`} 
              className="mri-layer base-layer" 
              alt="Base Scan" 
              draggable="false"
            />
            
            {/* AI Layer Logic */}
            {activeMode === 'split' ? (
               <>
                  <img 
                    src={`data:image/png;base64,${result.images.heatmap}`} 
                    className="mri-layer heatmap-layer" 
                    alt="Heatmap Split Layer"
                    draggable="false"
                    style={{ 
                       clipPath: `inset(0 0 0 ${sliderValue * 100}%)`,
                       opacity: 1 
                    }}
                  />
                  <div 
                    className="split-wipe-handle" 
                    style={{ left: `${sliderValue * 100}%` }}
                  />
               </>
            ) : (
               <img 
                 src={`data:image/png;base64,${result.images.heatmap}`} 
                 className="mri-layer heatmap-layer" 
                 alt="Heatmap Opacity Layer"
                 draggable="false"
                 style={{ opacity: activeMode === 'gradcam' ? sliderValue : 0 }}
               />
            )}
            
            {/* 2D TARGET HUD */}
            <TargetHUD location={selectedHotspot} />
          </motion.div>
        </div>
      ) : (
        /* ZERO-STATE DROPZONE */
        <div className="zero-state-dropzone">
          <input 
            type="file" 
            id="mri-upload" 
            accept="image/*" 
            onChange={handleFileUpload}
            hidden 
          />
          <label htmlFor="mri-upload" className="dropzone-area">
            {file ? (
              <div className="dropzone-content">
                <FileText size={48} className="accent-cyan" />
                <h2 className="dropzone-title">{file.name}</h2>
                <p className="dropzone-subtitle">Ready for analysis</p>
                <button 
                  className="workflow-btn glow float-btn"
                  onClick={(e) => { e.preventDefault(); analyzeMRI(); }}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? <Activity size={18} className="spin" /> : <Brain size={18} />}
                  {isAnalyzing ? 'PROCESSING...' : 'INITIATE INFERENCE'}
                </button>
              </div>
            ) : (
              <div className="dropzone-content">
                <Brain size={64} className="accent-cyan" style={{ opacity: 0.8 }} />
                <h2 className="dropzone-title">AWAITING INPUT</h2>
                <p className="dropzone-subtitle mt-2">Click or drag DICOM, JPG, PNG here</p>
              </div>
            )}
          </label>
        </div>
      )}

      {/* FLOATING TOP LEFT - BRAND & TELEMETRY */}
      <div className="hud-module top-left">
        <div className="brand-header minimal">
          <Brain size={24} className="accent-cyan" />
          <div className="brand-text">
            <h1>NEURAL NEXUS</h1>
            <p>Clinical Diagnostics HUD</p>
          </div>
        </div>

        <AnimatePresence>
          {result && (
            <motion.div 
              className="telemetry-minimal"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <div className="metric-row">
                <span className="metric-label">DIAGNOSIS</span>
                <span className="metric-value" style={{ color: result.label === 'No Tumor' ? 'var(--success)' : 'var(--danger)' }}>
                  {result.label !== 'No Tumor' ? result.label.toUpperCase() + ' DETECTED' : 'NO TUMOR'}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-label">CONFIDENCE</span>
                <span className="metric-value accent-cyan">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="metric-row">
                <span className="metric-label">ZOOM STATE</span>
                <span className="metric-value">{zoomLevel.toFixed(1)}x</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* 3D SPATIAL COPILOT (CINEMATIC PHYSICS WRAPPER) */}
      <AnimatePresence>
        {(isAnalyzing || result) && (
          <motion.div 
            layout
            className="spatial-canvas-wrapper"
            variants={spatialMapVariants}
            initial="docked"
            animate={isAnalyzing ? "global" : (isSpatialExpanded ? "expanded" : "docked")}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ type: "spring", stiffness: 200, damping: 25 }}
          >
            {!isAnalyzing && (
              <div className="copilot-header">
                 <Brain size={12} className="accent-cyan" /> 
                 <span>SPATIAL MAP</span>
                 <button 
                   className="expand-btn"
                   onClick={(e) => {
                     e.stopPropagation();
                     setIsSpatialExpanded(!isSpatialExpanded);
                   }}
                   title={isSpatialExpanded ? "Minimize" : "Expand"}
                 >
                   {isSpatialExpanded ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
                 </button>
              </div>
            )}
            <div className="canvas-container-placeholder" ref={constraintsRef}>
              <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
                <Suspense fallback={null}>
                  <BrainModel 
                    diagnosis={result?.label} 
                    tumorLocation={result?.tumor_location}
                    onHotspotClick={(loc) => setSelectedHotspot(loc)}
                    phase={!isAnalyzing ? 'docked' : (scanStatus.includes('INITIALIZING') ? 'dispersed' : 'forming')}
                    isDeconstructed={isDeconstructed}
                    isSpatialExpanded={isSpatialExpanded}
                  />
                </Suspense>
                <Preload all />
              </Canvas>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* FLOATING TOP RIGHT - EXPORT */}
      <AnimatePresence>
        {result && (
          <motion.div 
            className="hud-module top-right"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <button className="hud-action-btn" onClick={generateReport} disabled={isExporting}>
               <FileText size={16} /> 
               {isExporting ? 'GENERATING...' : 'EXPORT REPORT'}
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* FLOATING BOTTOM CENTER - MODES & ADJUSTER */}
      <AnimatePresence>
        {result && (
          <motion.div 
            className="hud-module bottom-center cinematic-hud"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
          >
            <div className="hud-pill-menu" style={{ maxWidth: '440px' }}>
              <button 
                className={activeMode === 'raw' ? 'active' : ''} 
                onClick={() => handleModeChange('raw')}
              >
                RAW
              </button>
              <button 
                className={activeMode === 'enhanced' ? 'active' : ''} 
                onClick={() => handleModeChange('enhanced')}
              >
                ENHANCED
              </button>
              <button 
                className={activeMode === 'split' ? 'active' : ''} 
                onClick={() => handleModeChange('split')}
              >
                SPLIT-VIEW
              </button>
              <button 
                className={activeMode === 'gradcam' ? 'active' : ''} 
                onClick={() => handleModeChange('gradcam')}
              >
                GRAD-CAM
              </button>
              <div className="hud-separator" />
              <button 
                className={`deconstruct-toggle ${isDeconstructed ? 'active accent-red' : ''}`}
                onClick={() => setIsDeconstructed(!isDeconstructed)}
                title="Deconstruct View"
              >
                {isDeconstructed ? 'RECONSTRUCT' : 'DECONSTRUCT'}
              </button>
            </div>

            <AnimatePresence>
              {(activeMode === 'gradcam' || activeMode === 'split') && (
                <motion.div 
                  className="hud-adjuster"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  style={{ overflow: 'hidden' }}
                >
                  <div style={{ padding: '0.5rem 0' }}>
                    <div className="adjuster-header">
                      <span className="adjuster-label">
                        {activeMode === 'split' ? 'WIPE BOUNDARY' : 'HEATMAP INTENSITY'}
                      </span>
                      <span className="adjuster-value">{Math.round(sliderValue * 100)}%</span>
                    </div>
                    <input 
                      type="range" 
                      className="styled-slider" 
                      min="0" max="1" step="0.01" 
                      value={sliderValue} 
                      onChange={(e) => setSliderValue(parseFloat(e.target.value))}
                      style={{ marginTop: '0.5rem' }}
                    />
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>

      {/* FLOATING BOTTOM RIGHT - NEXUS ORACLE */}
      <AnimatePresence>
        {result && (
          <div className="hud-module bottom-right">
            <RiskExplainerPanel />
          </div>
        )}
      </AnimatePresence>

      {/* OVERLAYS (Pulse Scan & PDF) */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div 
            className="scan-overlay fullscreen-override"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="pulse-grid"></div>
            <div className="scan-line"></div>
            <div className="scan-status-container">
              <Activity size={32} className="spin accent-cyan" />
              <h2>{scanStatus}</h2>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {pdfPreviewUrl && (
          <motion.div 
            className="preview-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div 
              className="preview-modal"
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
            >
              <div className="preview-header">
                <div className="preview-title">
                  <FileText size={18} className="accent-cyan-text" />
                  CLINICAL REPORT PREVIEW
                </div>
                <div className="preview-actions">
                  <button className="action-btn close" onClick={closePreview}>
                    CANCEL
                  </button>
                  <button className="action-btn download" onClick={downloadPDF}>
                    <Download size={14} /> DOWNLOAD DOCUMENT
                  </button>
                </div>
              </div>
              <iframe src={pdfPreviewUrl} className="preview-iframe" title="PDF Preview" />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}

export default App;
