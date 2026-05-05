'use client';

import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useCaseStore } from '@/store/caseStore';
import { useToastStore } from '@/store/toastStore';
import { PredictionSummary } from '@/components/case/viewer/PredictionSummary';
import { GradCamViewer } from '@/components/case/viewer/GradCamViewer';
import { UploadModal } from '@/components/case/UploadModal';
import { CXRSkeleton, PredictionSkeleton } from '@/components/shared/Skeleton';
import { reinferCase, regenerateGradCam, flagCaseCritical, requestReanalysis } from '@/lib/api';
import { useUserRole } from '@/lib/userRole';
import {
  Layers, Eye, EyeOff, Settings2, SlidersHorizontal, RefreshCcw,
  MonitorPlay, Columns, GalleryHorizontalEnd, RotateCcw,
  PanelRightClose, PanelRightOpen, Flame, CheckCircle2, XCircle, ZoomIn, ZoomOut, Maximize2,
  Upload, ImageIcon, RefreshCw, MessageSquare, Lock, AlertTriangle,
} from 'lucide-react';
import Image from 'next/image';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

export function CXRAnalysisTab() {
  const currentCase = useCaseStore((state) => state.currentCase);
  const refreshCurrentCase = useCaseStore((state) => state.refreshCurrentCase);
  const selectedLabel = useCaseStore((state) => state.selectedLabel);
  const setSelectedLabel = useCaseStore((state) => state.setSelectedLabel);
  const addToast = useToastStore((state) => state.addToast);

  const showOverlay = useCaseStore((state) => state.showOverlay);
  const toggleOverlay = useCaseStore((state) => state.toggleOverlay);

  const setBrightness = useCaseStore((state) => state.setBrightness);
  const setContrast = useCaseStore((state) => state.setContrast);
  const setZoom = useCaseStore((state) => state.setZoom);
  const resetViewport = useCaseStore((state) => state.resetViewport);
  const panOffset = useCaseStore((state) => state.panOffset);
  const setPanOffset = useCaseStore((state) => state.setPanOffset);

  const cxrViewMode = useCaseStore((state) => state.cxrViewMode);
  const setCxrViewMode = useCaseStore((state) => state.setCxrViewMode);

  const [isReinferring, setIsReinferring] = useState(false);
  const [isRegeneratingGradCam, setIsRegeneratingGradCam] = useState(false);
  const [isPredictionPanelOpen, setIsPredictionPanelOpen] = useState(true);
  const [showUploadModal, setShowUploadModal] = useState(false);

  // Role gating: only the radiologist owns CXR. Ward doctors get a read-only
  // viewer + a "Request Reanalysis" stub (per spec: "no controls for ward").
  const { role } = useUserRole();
  const canModifyCxr = role === 'radiologist';

  const [isRequestingReanalysis, setIsRequestingReanalysis] = useState(false);
  const handleRequestReanalysis = async () => {
    if (!currentCase) return;
    setIsRequestingReanalysis(true);
    try {
      await requestReanalysis(
        currentCase.case.id,
        `Ward doctor requests reanalysis for ${currentCase.patient.first_name} ${currentCase.patient.last_name}.`,
      );
      addToast({
        type: 'success',
        title: 'Reanalysis requested',
        message: 'Radiologist will see this case flagged in their queue.',
      });
      await refreshCurrentCase();
    } catch (err: any) {
      addToast({
        type: 'error',
        title: 'Could not send request',
        message: err?.message || 'Try again or contact admin.',
      });
    } finally {
      setIsRequestingReanalysis(false);
    }
  };

  const [isFlagging, setIsFlagging] = useState(false);
  const handleFlagCritical = async () => {
    if (!currentCase) return;
    const finding = selectedLabel || currentCase.case.cxr_heatmap_label || 'Unspecified';
    setIsFlagging(true);
    try {
      await flagCaseCritical(currentCase.case.id, finding);
      addToast({
        type: 'success',
        title: 'Flagged critical',
        message: `Ward doctor notified. Finding: ${finding}.`,
      });
      await refreshCurrentCase();
    } catch (err: any) {
      addToast({ type: 'error', title: 'Flag failed', message: err.message || 'Could not flag case.' });
    } finally {
      setIsFlagging(false);
    }
  };

  // Pan interaction for side-by-side view
  const isPanning = useRef(false);
  const lastPos = useRef({ x: 0, y: 0 });
  const sbsContainerRef = useRef<HTMLDivElement>(null);

  // Auto-select the finding with highest probability if nothing is selected
  useEffect(() => {
    if (currentCase && !selectedLabel && currentCase.predictions.length > 0) {
      const topFinding = [...currentCase.predictions].sort((a, b) => b.probability - a.probability)[0];
      setSelectedLabel(topFinding.label);
    }
  }, [currentCase, selectedLabel, setSelectedLabel]);

  const handleReinfer = async (e?: React.MouseEvent) => {
    e?.preventDefault();
    e?.stopPropagation();
    if (!currentCase) return;
    setIsReinferring(true);
    try {
      await reinferCase(currentCase.case.id);
      addToast({ type: 'info', title: 'Inference Queued', message: 'Re-running full CXR analysis. Results will appear shortly.' });
      setTimeout(async () => {
        await refreshCurrentCase();
        setIsReinferring(false);
      }, 5000);
    } catch (err: any) {
      setIsReinferring(false);
      addToast({ type: 'error', title: 'Inference Failed', message: err.message || 'Could not re-run inference.' });
    }
  };

  const handleRegenerateGradCam = async (label?: string) => {
    if (!currentCase) return;
    const targetLabel = label ?? selectedLabel;
    if (!targetLabel) return;
    setIsRegeneratingGradCam(true);
    try {
      await regenerateGradCam(currentCase.case.id, targetLabel);
      addToast({ type: 'info', title: 'Grad-CAM Queued', message: `Generating heatmap for "${targetLabel}"…` });
      setTimeout(async () => {
        await refreshCurrentCase();
        setIsRegeneratingGradCam(false);
      }, 6000);
    } catch (err: any) {
      setIsRegeneratingGradCam(false);
      addToast({ type: 'error', title: 'Grad-CAM Failed', message: err.message || 'Could not regenerate heatmap.' });
    }
  };

  // Auto-generate Grad-CAM when switching to a label that has no heatmap yet
  const prevLabelRef = useRef<string | null>(null);
  useEffect(() => {
    if (!selectedLabel || !currentCase || isRegeneratingGradCam) return;
    if (selectedLabel === prevLabelRef.current) return;
    prevLabelRef.current = selectedLabel;
    const pred = currentCase.predictions.find(p => p.label === selectedLabel);
    if (pred && !pred.gradcam_url) {
      handleRegenerateGradCam(selectedLabel);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedLabel, currentCase]);

  // Side-by-side pan/zoom handlers (shared with Zustand viewport state)
  const handleSbsPointerDown = (e: React.PointerEvent) => {
    isPanning.current = true;
    lastPos.current = { x: e.clientX, y: e.clientY };
    if (e.target instanceof Element) e.target.setPointerCapture(e.pointerId);
  };

  const handleSbsPointerMove = useCallback((e: React.PointerEvent) => {
    if (!isPanning.current) return;
    const dx = e.clientX - lastPos.current.x;
    const dy = e.clientY - lastPos.current.y;
    const zoom = useCaseStore.getState().currentCase?.consultation?.viewport_state?.zoom ?? 1;
    setPanOffset(prev => ({ x: prev.x + dx / zoom, y: prev.y + dy / zoom }));
    lastPos.current = { x: e.clientX, y: e.clientY };
  }, [setPanOffset]);

  const handleSbsPointerUp = (e: React.PointerEvent) => {
    isPanning.current = false;
    if (e.target instanceof Element && e.target.hasPointerCapture(e.pointerId)) {
      e.target.releasePointerCapture(e.pointerId);
    }
  };

  const handleSbsWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.15 : 0.15;
    const current = useCaseStore.getState().currentCase?.consultation?.viewport_state?.zoom ?? 1;
    setZoom(Math.max(0.5, Math.min(5, current + delta)));
  };

  if (!currentCase) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        No case selected.
      </div>
    );
  }

  const predictions = currentCase.predictions;
  const viewportState = currentCase.consultation?.viewport_state as any;

  const brightnessValue = viewportState?.brightness ?? 100;
  const contrastValue = viewportState?.contrast ?? 100;
  const zoomValue = viewportState?.zoom ?? 1;

  const activePrediction = predictions.find(p => p.label === selectedLabel);
  const hasBaseImage = !!currentCase.case.cxr_dicom_url;
  const baseImage = currentCase.case.cxr_dicom_url || '';
  const hasPredictions = predictions.some(p => p.probability > 0);
  const inferenceLoading = hasBaseImage && !hasPredictions;
  const heatmapImage = activePrediction?.gradcam_url;

  const transformStyle = `scale(${zoomValue}) translate(${panOffset.x}px, ${panOffset.y}px)`;
  const filterStyle = `brightness(${brightnessValue}%) contrast(${contrastValue}%)`;

  return (
    <div className="h-full w-full max-w-[1400px] mx-auto flex flex-col gap-6">

      {/* No-CXR prominent upload prompt (radiologist-only). Ward sees a wait state. */}
      {!hasBaseImage && canModifyCxr && (
        <div className="bg-gradient-to-br from-blue-50/60 via-white to-slate-50 border border-blue-100 rounded-2xl shadow-card p-8 flex flex-col items-center justify-center text-center min-h-[420px] animate-fadeInUp">
          <div className="relative">
            <div className="absolute inset-0 bg-blue-200/40 rounded-full blur-2xl animate-pulseSoft" />
            <div className="relative bg-white border border-blue-200 rounded-2xl p-5 shadow-md mb-5">
              <ImageIcon className="w-10 h-10 text-blue-600" />
            </div>
          </div>
          <h3 className="text-lg font-bold text-slate-900 mb-1">No chest X-ray on file</h3>
          <p className="text-sm text-slate-500 max-w-md mb-6">
            Upload a CXR image to run DenseNet121 inference, generate Grad-CAM heatmaps, and find similar cases via FAISS.
          </p>
          <button
            onClick={() => setShowUploadModal(true)}
            className="group inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white text-sm font-bold rounded-xl shadow-lg shadow-blue-500/25 hover:shadow-xl hover:shadow-blue-500/35 hover:-translate-y-0.5 transition-all duration-200"
          >
            <Upload className="w-4 h-4 transition-transform group-hover:-translate-y-0.5" />
            Upload CXR Image
          </button>
          <p className="text-[11px] text-slate-400 mt-4 font-mono">PNG · JPEG · DICOM — max 10 MB</p>
        </div>
      )}

      {/* Ward Doctor view: no CXR yet → waiting on radiologist */}
      {!hasBaseImage && !canModifyCxr && (
        <div className="bg-white border border-slate-200 rounded-2xl shadow-card p-8 flex flex-col items-center justify-center text-center min-h-[300px] animate-fadeInUp">
          <div className="w-12 h-12 rounded-2xl bg-amber-50 flex items-center justify-center mb-4">
            <ImageIcon className="w-6 h-6 text-amber-500" />
          </div>
          <h3 className="text-base font-semibold text-slate-900 mb-1">Awaiting radiologist</h3>
          <p className="text-sm text-slate-500 max-w-md">
            No chest X-ray has been uploaded yet. The radiologist on duty will attach it and run analysis.
          </p>
        </div>
      )}

      {/* Toolbar */}
      {hasBaseImage && (
        <div className="flex items-center justify-between bg-white border border-gray-100 rounded-xl px-4 py-2.5 shadow-card flex-wrap gap-2 animate-fadeInDown">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className={cn(
              "w-1.5 h-1.5 rounded-full",
              hasPredictions ? "bg-emerald-500" : "bg-amber-400 animate-pulseSoft"
            )} />
            {hasPredictions
              ? <><span className="font-semibold text-gray-700">{predictions.filter(p => p.probability > 0).length}</span> findings analysed · MC Dropout: <span className="font-mono">{predictions[0]?.mc_passes ?? 0}</span> passes</>
              : 'Inference pending — results will appear automatically'}
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            <button
              onClick={() => setIsPredictionPanelOpen(v => !v)}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-200 rounded-lg text-xs font-semibold text-gray-600 hover:border-gray-300 hover:bg-gray-50 transition-all duration-150"
            >
              {isPredictionPanelOpen ? <PanelRightClose className="w-3.5 h-3.5" /> : <PanelRightOpen className="w-3.5 h-3.5" />}
              {isPredictionPanelOpen ? 'Hide Panel' : 'Show Panel'}
            </button>
            {canModifyCxr && (
              <>
                <button
                  onClick={() => setShowUploadModal(true)}
                  className="group flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-200 rounded-lg text-xs font-semibold text-gray-700 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 transition-all duration-150"
                  title="Replace the current CXR image"
                >
                  <RefreshCw className="w-3.5 h-3.5 transition-transform group-hover:rotate-180 duration-300" />
                  Replace Image
                </button>
                <button
                  onClick={() => handleRegenerateGradCam()}
                  disabled={isRegeneratingGradCam || !selectedLabel}
                  className="group flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-200 rounded-lg text-xs font-semibold text-gray-700 hover:border-orange-300 hover:bg-orange-50 hover:text-orange-700 transition-all duration-150 disabled:opacity-50"
                  title={selectedLabel ? `Regenerate Grad-CAM for "${selectedLabel}"` : 'Select a finding first'}
                >
                  <Flame className={cn("w-3.5 h-3.5 transition-colors", isRegeneratingGradCam ? "animate-pulse text-orange-500" : "group-hover:text-orange-600")} />
                  {isRegeneratingGradCam ? 'Generating…' : 'Rerun Grad-CAM'}
                </button>
                <button
                  onClick={handleReinfer}
                  disabled={isReinferring}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold shadow-sm transition-all duration-150 disabled:opacity-50"
                >
                  <RotateCcw className={cn("w-3.5 h-3.5", isReinferring && "animate-spin")} />
                  {isReinferring ? 'Re-running…' : 'Re-run Inference'}
                </button>
                <button
                  onClick={handleFlagCritical}
                  disabled={isFlagging}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white rounded-lg text-xs font-semibold shadow-sm transition-all duration-150 disabled:opacity-50"
                  title="Notify ward doctor of a critical finding"
                >
                  <AlertTriangle className={cn("w-3.5 h-3.5", isFlagging && "animate-pulse")} />
                  {isFlagging ? 'Flagging…' : 'Flag Critical'}
                </button>
              </>
            )}

            {!canModifyCxr && (
              <>
                <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-100 text-slate-500 text-[11px] font-medium">
                  <Lock className="w-3 h-3" />
                  Read-only
                </span>
                <button
                  onClick={handleRequestReanalysis}
                  disabled={isRequestingReanalysis}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-semibold shadow-sm transition-all duration-150 disabled:opacity-50"
                  title="Notify the radiologist to re-review this CXR"
                >
                  <MessageSquare className={cn("w-3.5 h-3.5", isRequestingReanalysis && "animate-pulse")} />
                  {isRequestingReanalysis ? 'Sending…' : 'Request Reanalysis'}
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {hasBaseImage && (
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full min-h-[600px]">

        {/* Left Column: Controls */}
        <div className={cn("flex flex-col gap-4 overflow-y-auto pr-2 pb-4 lg:col-span-3")}>

          {/* View Mode Toggle */}
          <div className="bg-white border border-gray-100 rounded-xl shadow-card p-5 animate-fadeInUp stagger-0">
            <div className="flex items-center justify-between mb-4 border-b border-gray-100 pb-3">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-md bg-blue-50 text-blue-600 flex items-center justify-center">
                  <Layers className="w-3.5 h-3.5" />
                </div>
                <h3 className="font-semibold text-gray-900 text-sm">View Mode</h3>
              </div>
              <button
                onClick={toggleOverlay}
                disabled={!selectedLabel}
                className={cn(
                  "p-1.5 rounded-lg transition-all duration-150",
                  showOverlay
                    ? "bg-blue-600 text-white hover:bg-blue-700 shadow-sm"
                    : "bg-gray-100 text-gray-500 hover:bg-gray-200"
                )}
                title={showOverlay ? "Hide AI Overlay" : "Show AI Overlay"}
              >
                {showOverlay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
              </button>
            </div>
            <div className="flex gap-2 bg-gray-50 rounded-lg p-1">
              <button
                onClick={() => setCxrViewMode('curtain')}
                className={cn(
                  "flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-semibold transition-all duration-150",
                  cxrViewMode === 'curtain'
                    ? "bg-white text-blue-700 shadow-sm ring-1 ring-gray-200"
                    : "text-gray-600 hover:text-gray-900"
                )}
              >
                <GalleryHorizontalEnd className="w-3.5 h-3.5" />
                Curtain
              </button>
              <button
                onClick={() => setCxrViewMode('side-by-side')}
                className={cn(
                  "flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-semibold transition-all duration-150",
                  cxrViewMode === 'side-by-side'
                    ? "bg-white text-blue-700 shadow-sm ring-1 ring-gray-200"
                    : "text-gray-600 hover:text-gray-900"
                )}
              >
                <Columns className="w-3.5 h-3.5" />
                Split
              </button>
            </div>
          </div>

          {/* Enhancements */}
          <div className="bg-white border border-gray-100 rounded-xl shadow-card p-5 flex flex-col animate-fadeInUp stagger-1">
            <div className="flex items-center justify-between mb-5 border-b border-gray-100 pb-3">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded-md bg-slate-100 text-slate-600 flex items-center justify-center">
                  <Settings2 className="w-3.5 h-3.5" />
                </div>
                <h3 className="font-semibold text-gray-900 text-sm">Enhancements</h3>
              </div>
              <button
                onClick={resetViewport}
                className="flex items-center gap-1.5 px-2 py-1 text-gray-500 hover:text-gray-900 hover:bg-gray-100 rounded-md text-xs font-medium transition-colors"
              >
                <RefreshCcw className="w-3 h-3" /> Reset
              </button>
            </div>
            <div className="space-y-5">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Brightness</label>
                  <span className="text-xs text-gray-600 font-mono tabular-nums">{brightnessValue}%</span>
                </div>
                <input type="range" min="0" max="200" value={brightnessValue} onChange={(e) => setBrightness(parseInt(e.target.value))} className="w-full accent-blue-600 cursor-pointer" />
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Contrast</label>
                  <span className="text-xs text-gray-600 font-mono tabular-nums">{contrastValue}%</span>
                </div>
                <input type="range" min="0" max="200" value={contrastValue} onChange={(e) => setContrast(parseInt(e.target.value))} className="w-full accent-blue-600 cursor-pointer" />
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">Zoom</label>
                  <span className="text-xs text-gray-600 font-mono tabular-nums">{zoomValue.toFixed(1)}x</span>
                </div>
                <input type="range" min="0.5" max="5" step="0.1" value={zoomValue} onChange={(e) => setZoom(parseFloat(e.target.value))} className="w-full accent-blue-600 cursor-pointer" />
              </div>
            </div>
          </div>

          {/* Active Finding Selector */}
          <div className="bg-white border border-gray-100 rounded-xl shadow-card p-5 flex-1 animate-fadeInUp stagger-2">
            <div className="flex items-center gap-2 mb-4 border-b border-gray-100 pb-3">
              <div className="w-6 h-6 rounded-md bg-purple-50 text-purple-600 flex items-center justify-center">
                <SlidersHorizontal className="w-3.5 h-3.5" />
              </div>
              <h3 className="font-semibold text-gray-900 text-sm">Active Finding</h3>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {predictions.map(pred => {
                const isSelected = selectedLabel === pred.label;
                const hasHeatmap = !!pred.gradcam_url;
                return (
                  <button
                    key={pred.id}
                    onClick={() => setSelectedLabel(pred.label)}
                    className={cn(
                      "group px-3 py-1.5 rounded-full text-xs font-semibold border transition-all duration-150 flex items-center gap-1.5",
                      isSelected
                        ? "bg-blue-600 border-blue-600 text-white shadow-sm"
                        : "bg-white border-gray-200 text-gray-600 hover:border-blue-300 hover:text-blue-700 hover:bg-blue-50"
                    )}
                  >
                    {pred.label}
                    {isSelected && (
                      hasHeatmap
                        ? <CheckCircle2 className="w-3 h-3 text-white shrink-0" />
                        : <XCircle className="w-3 h-3 text-white/70 shrink-0" aria-label="No heatmap yet" />
                    )}
                  </button>
                );
              })}
            </div>
            {selectedLabel && !heatmapImage && (
              <div className="flex items-start gap-2 text-[11px] text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 mt-3 leading-relaxed">
                <Flame className="w-3 h-3 mt-0.5 shrink-0" />
                <span>No heatmap for <strong>{selectedLabel}</strong> yet. Click <strong>Rerun Grad-CAM</strong> above.</span>
              </div>
            )}
          </div>

          {/* Viewport Data */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl shadow-sm p-4 mt-auto">
            <div className="flex items-center gap-2 mb-3">
              <MonitorPlay className="w-4 h-4 text-slate-500" />
              <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">Viewport Data</h4>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-white border border-slate-200 p-2 rounded flex flex-col">
                <span className="text-slate-400 font-medium mb-0.5">Zoom Level</span>
                <span className="font-mono text-slate-700 font-semibold">{zoomValue.toFixed(2)}x</span>
              </div>
              <div className="bg-white border border-slate-200 p-2 rounded flex flex-col">
                <span className="text-slate-400 font-medium mb-0.5">Pan (X, Y)</span>
                <span className="font-mono text-slate-700 font-semibold">{Math.round(panOffset.x)}, {Math.round(panOffset.y)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Middle Column: Image Viewer */}
        <div className={cn("h-[500px] lg:h-full", isPredictionPanelOpen ? "lg:col-span-6" : "lg:col-span-9")}>
          {cxrViewMode === 'curtain' ? (
            <GradCamViewer />
          ) : (
            /* ── Side-by-Side View with synchronized pan/zoom ── */
            <div
              ref={sbsContainerRef}
              className="h-full bg-[#0f172a] rounded-xl border border-gray-800 overflow-hidden flex flex-col"
            >
              {/* Zoom toolbar */}
              <div className="flex items-center justify-between px-4 py-2 bg-slate-900/80 border-b border-slate-700 shrink-0">
                <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-widest">Split View — Synchronized Pan &amp; Zoom</span>
                <div className="flex items-center gap-1.5">
                  <button onClick={() => setZoom(Math.max(0.5, zoomValue - 0.2))} className="p-1 bg-slate-700 hover:bg-slate-600 rounded text-white transition-colors">
                    <ZoomOut className="w-3 h-3" />
                  </button>
                  <span className="text-[10px] text-slate-300 font-mono w-10 text-center">{zoomValue.toFixed(1)}x</span>
                  <button onClick={() => setZoom(Math.min(5, zoomValue + 0.2))} className="p-1 bg-slate-700 hover:bg-slate-600 rounded text-white transition-colors">
                    <ZoomIn className="w-3 h-3" />
                  </button>
                  <button onClick={() => { setZoom(1); setPanOffset({ x: 0, y: 0 }); }} className="p-1 bg-slate-700 hover:bg-slate-600 rounded text-white transition-colors" title="Reset (0)">
                    <Maximize2 className="w-3 h-3" />
                  </button>
                </div>
              </div>

              {/* Two panels */}
              <div
                className="flex-1 flex gap-[2px] overflow-hidden cursor-grab active:cursor-grabbing select-none touch-none"
                onPointerDown={handleSbsPointerDown}
                onPointerMove={handleSbsPointerMove}
                onPointerUp={handleSbsPointerUp}
                onPointerCancel={handleSbsPointerUp}
                onWheel={handleSbsWheel}
              >
                {/* Left: Raw CXR */}
                <div className="flex-1 relative bg-[#020617] overflow-hidden">
                  <div className="absolute top-2 left-2 z-10 bg-black/60 text-white text-[10px] font-semibold px-2 py-1 rounded backdrop-blur-sm border border-white/10">
                    ORIGINAL CXR
                  </div>
                  <div
                    className="absolute inset-0 origin-center transition-transform duration-75 ease-linear"
                    style={{ transform: transformStyle, filter: filterStyle }}
                  >
                    {hasBaseImage ? (
                      <Image
                        src={baseImage}
                        alt="Original CXR"
                        fill
                        unoptimized
                        draggable={false}
                        className="object-contain pointer-events-none"
                      />
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-xs">No CXR available</div>
                    )}
                  </div>
                </div>

                {/* Divider */}
                <div className="w-[2px] bg-slate-600 shrink-0" />

                {/* Right: Grad-CAM */}
                <div className="flex-1 relative bg-[#020617] overflow-hidden">
                  <div className="absolute top-2 left-2 z-10 bg-black/60 text-white text-[10px] font-semibold px-2 py-1 rounded backdrop-blur-sm border border-white/10">
                    GRAD-CAM: {selectedLabel || 'None selected'}
                  </div>
                  <div
                    className="absolute inset-0 origin-center transition-transform duration-75 ease-linear"
                    style={{ transform: transformStyle }}
                  >
                    {heatmapImage ? (
                      <Image
                        src={heatmapImage}
                        alt={`Grad-CAM: ${selectedLabel}`}
                        fill
                        unoptimized
                        draggable={false}
                        className="object-contain pointer-events-none"
                      />
                    ) : (
                      <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 gap-3 text-center px-6">
                        <Flame className="w-8 h-8 text-orange-500/40" aria-hidden="true" />
                        <div>
                          <p className="text-sm font-semibold text-slate-300">No Grad-CAM Available</p>
                          <p className="text-[11px] text-slate-500 mt-1">
                            {selectedLabel
                              ? `Click "Rerun Grad-CAM" to generate heatmap for "${selectedLabel}"`
                              : 'Select a finding from the left panel'}
                          </p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Footer hint */}
              <div className="px-4 py-1.5 bg-slate-900 border-t border-slate-800 flex items-center gap-4 shrink-0">
                {[['Drag', 'Pan both views'], ['Scroll', 'Zoom'], ['Reset btn', 'Reset view']].map(([k, d]) => (
                  <div key={k} className="flex items-center gap-1">
                    <kbd className="text-[9px] bg-slate-700 text-slate-300 px-1.5 py-0.5 rounded font-mono">{k}</kbd>
                    <span className="text-[10px] text-slate-500">{d}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Prediction Summary */}
        {isPredictionPanelOpen && (
          <div className="lg:col-span-3 h-[400px] lg:h-full">
            {inferenceLoading || isReinferring ? <PredictionSkeleton /> : <PredictionSummary />}
          </div>
        )}

      </div>
      )}

      {/* Upload modal — wires into existing /api/cases/{id}/upload/cxr endpoint */}
      {showUploadModal && (
        <UploadModal
          type="cxr"
          caseId={currentCase.case.id}
          onClose={() => setShowUploadModal(false)}
        />
      )}
    </div>
  );
}
