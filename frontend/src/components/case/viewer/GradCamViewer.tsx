'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import Image from 'next/image';
import { useCaseStore } from '@/store/caseStore';
import { Maximize2, ZoomIn, ZoomOut, GripVertical, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

export function GradCamViewer() {
  const currentCase = useCaseStore((state) => state.currentCase);
  const selectedLabel = useCaseStore((state) => state.selectedLabel);
  const showOverlay = useCaseStore((state) => state.showOverlay);
  const setZoom = useCaseStore((state) => state.setZoom);
  
  const sliderPosition = useCaseStore((state) => state.sliderPosition);
  const setSliderPosition = useCaseStore((state) => state.setSliderPosition);
  
  const panOffset = useCaseStore((state) => state.panOffset);
  const setPanOffset = useCaseStore((state) => state.setPanOffset);

  const [baseImageError, setBaseImageError] = useState(false);
  const [overlayImageError, setOverlayImageError] = useState(false);
  
  // Reset overlay error when the user selects a different label
  useEffect(() => {
    setOverlayImageError(false);
  }, [selectedLabel]);
  
  const containerRef = useRef<HTMLDivElement>(null);
  
  // Interaction states
  const isDraggingSlider = useRef(false);
  const isPanning = useRef(false);
  const lastPos = useRef({ x: 0, y: 0 });

  const handlePointerDownSlider = (e: React.PointerEvent | React.TouchEvent) => {
    e.stopPropagation();
    isDraggingSlider.current = true;
  };

  const handlePointerDownContainer = (e: React.PointerEvent) => {
    isPanning.current = true;
    lastPos.current = { x: e.clientX, y: e.clientY };
    // Optional: capture pointer to continue drag outside
    if (e.target instanceof Element) {
      e.target.setPointerCapture(e.pointerId);
    }
  };

  const handleTouchStartContainer = (e: React.TouchEvent) => {
    isPanning.current = true;
    lastPos.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
  };

  const handlePointerUp = (e: React.PointerEvent | React.TouchEvent) => {
    isDraggingSlider.current = false;
    isPanning.current = false;
    if ('pointerId' in e && e.target instanceof Element && e.target.hasPointerCapture(e.pointerId)) {
      e.target.releasePointerCapture(e.pointerId);
    }
  };

  const handleMove = useCallback((clientX: number, clientY: number) => {
    if (isDraggingSlider.current && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      let x = clientX - rect.left;
      x = Math.max(0, Math.min(x, rect.width));
      setSliderPosition((x / rect.width) * 100);
    } else if (isPanning.current) {
      const dx = clientX - lastPos.current.x;
      const dy = clientY - lastPos.current.y;
      
      // We must divide by zoom to ensure the mouse stays exactly on the point it clicked
      const zoomValue = useCaseStore.getState().currentCase?.consultation?.viewport_state?.zoom ?? 1;
      
      setPanOffset(prev => ({
        x: prev.x + (dx / zoomValue),
        y: prev.y + (dy / zoomValue)
      }));
      
      lastPos.current = { x: clientX, y: clientY };
    }
  }, [setSliderPosition, setPanOffset]);

  const onPointerMove = (e: React.PointerEvent) => handleMove(e.clientX, e.clientY);
  const onTouchMove = (e: React.TouchEvent) => handleMove(e.touches[0].clientX, e.touches[0].clientY);

  if (!currentCase) return null;

  const { case: caseData, predictions, consultation } = currentCase;
  const activePrediction = predictions.find(p => p.label === selectedLabel);
  
  const hasImage = !!caseData.cxr_dicom_url;
  const baseImage = caseData.cxr_dicom_url || '';
  const heatmapOverlay = activePrediction?.gradcam_url;
  
  const viewportState = consultation?.viewport_state;
  const brightness = viewportState?.brightness ?? 100;
  const contrast = viewportState?.contrast ?? 100;
  const zoom = viewportState?.zoom ?? 1;

  // Visual logic
  const transformStyle = `scale(${zoom}) translate(${panOffset.x}px, ${panOffset.y}px)`;
  const filterStyle = `brightness(${brightness}%) contrast(${contrast}%)`;
  const clipPathStyle = `inset(0 ${100 - sliderPosition}% 0 0)`;
  
  const isGrabbing = isPanning.current;

  return (
    <div className="bg-[#0f172a] border border-gray-800 rounded-xl shadow-inner h-full flex flex-col overflow-hidden relative group">
      
      {/* Viewer Toolbar */}
      <div className="absolute top-0 inset-x-0 z-30 bg-gradient-to-b from-black/80 to-transparent p-4 flex justify-between items-start opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none">
        <div className="text-white pointer-events-auto">
          <div className="text-sm font-semibold tracking-wide drop-shadow-md">
            {activePrediction && showOverlay && heatmapOverlay
              ? `Grad-CAM: ${activePrediction.label}`
              : activePrediction
              ? `Raw DICOM — ${activePrediction.label} selected`
              : 'Raw DICOM View'}
          </div>
        </div>
        <div className="flex items-center gap-1.5 pointer-events-auto">
          <button onClick={() => setZoom(Math.max(0.5, zoom - 0.2))} className="p-1.5 bg-black/60 hover:bg-white/20 rounded text-white backdrop-blur-sm transition-colors border border-white/10">
            <ZoomOut className="w-4 h-4" />
          </button>
          <button onClick={() => setZoom(Math.min(5, zoom + 0.2))} className="p-1.5 bg-black/60 hover:bg-white/20 rounded text-white backdrop-blur-sm transition-colors border border-white/10">
            <ZoomIn className="w-4 h-4" />
          </button>
          <div className="w-px h-5 bg-white/20 mx-1" />
          <button onClick={() => { setZoom(1); setPanOffset({x:0, y:0}); }} className="p-1.5 bg-black/60 hover:bg-white/20 rounded text-white backdrop-blur-sm transition-colors border border-white/10">
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Main Image Stage */}
      <div className="flex-1 relative bg-[#020617] flex items-center justify-center p-2 overflow-hidden">
        
        {/* Interactive Container */}
        <div 
          ref={containerRef}
          className={cn(
            "relative w-full h-full overflow-hidden rounded-lg border border-slate-700 bg-slate-900 select-none touch-none",
            isGrabbing ? "cursor-grabbing" : "cursor-grab"
          )}
          onPointerDown={handlePointerDownContainer}
          onPointerMove={onPointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          onTouchStart={handleTouchStartContainer}
          onTouchMove={onTouchMove}
          onTouchEnd={handlePointerUp}
          onTouchCancel={handlePointerUp}
        >
          {/* Zoom & Pan Wrapper */}
          <div 
            className="absolute inset-0 origin-center transition-transform duration-75 ease-linear"
            style={{ 
              transform: transformStyle,
              filter: filterStyle
            }}
          >
            {/* Base Layer: Raw CXR */}
            <div className="absolute inset-0 z-0">
              {hasImage && !baseImageError ? (
                <Image 
                  src={baseImage} 
                  alt="Raw CXR Base Layer" 
                  fill
                  priority
                  unoptimized
                  draggable={false}
                  className="object-contain pointer-events-none"
                  onError={() => setBaseImageError(true)}
                />
              ) : hasImage && baseImageError ? (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
                  <AlertCircle className="w-8 h-8 mb-2 opacity-50" />
                  <span className="text-xs font-mono">{baseImage}</span>
                  <span className="text-[10px] text-slate-400 mt-1">Image failed to load</span>
                </div>
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 gap-3">
                  <div className="w-16 h-16 rounded-xl border-2 border-dashed border-slate-600 flex items-center justify-center">
                    <AlertCircle className="w-8 h-8 opacity-40" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-semibold text-slate-300">No CXR Available</p>
                    <p className="text-[10px] text-slate-500 mt-1">Upload a chest X-ray via the Ingestion Wizard</p>
                  </div>
                </div>
              )}
            </div>
            
            {/* Heatmap Overlay */}
            {heatmapOverlay && showOverlay && (
              <div 
                className="absolute inset-0 z-10 pointer-events-none"
                style={{ clipPath: clipPathStyle }}
              >
                {!overlayImageError ? (
                  <Image 
                    src={heatmapOverlay} 
                    alt={`Heatmap`} 
                    fill
                    priority
                    unoptimized
                    draggable={false}
                    className="object-contain"
                    style={{ mixBlendMode: 'multiply' }}
                    onError={() => setOverlayImageError(true)}
                  />
                ) : null}
              </div>
            )}
          </div>
          
          {/* Draggable Divider Line (Outside of Zoom Transform, stays positioned relative to container) */}
          {heatmapOverlay && showOverlay && (
            <div 
              className="absolute top-0 bottom-0 z-20 w-[2px] bg-white cursor-col-resize hover:w-[4px] transition-all shadow-[0_0_10px_rgba(0,0,0,0.8)]"
              style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
              onPointerDown={handlePointerDownSlider}
              onTouchStart={handlePointerDownSlider}
            >
              {/* Custom Handle */}
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-white text-blue-600 rounded-full p-1.5 shadow-xl border border-blue-200">
                <GripVertical className="w-4 h-4" />
              </div>
            </div>
          )}

        </div>
      </div>
      
      {/* Information Footer */}
      {activePrediction && (
        <div className="absolute bottom-0 inset-x-0 z-20 bg-gradient-to-t from-black/90 via-black/60 to-transparent p-4 pt-12 flex flex-col justify-end pointer-events-none">
          {activePrediction.uncertainty_level ? (
            /* ── MC Dropout data available ── */
            <div className="text-white flex items-end justify-between pointer-events-auto">
              <div className="flex flex-col gap-1">
                <span className="text-[10px] text-gray-400 uppercase tracking-widest font-semibold">Uncertainty Estimate</span>
                <span className={cn(
                  "text-sm font-medium px-2 py-0.5 rounded border inline-block w-fit",
                  activePrediction.uncertainty_level === 'High Uncertainty' ? "bg-red-900/50 border-red-500 text-red-200" :
                  activePrediction.uncertainty_level === 'Moderate Uncertainty' ? "bg-amber-900/50 border-amber-500 text-amber-200" :
                  "bg-emerald-900/50 border-emerald-500 text-emerald-200"
                )}>
                  {activePrediction.uncertainty_level}
                </span>
              </div>
              <div className="text-right flex flex-col gap-1">
                <span className="text-[10px] text-gray-400 uppercase tracking-widest font-semibold">MC Variance</span>
                <span className="text-sm font-medium font-mono text-blue-200 bg-blue-900/30 px-2 py-0.5 rounded border border-blue-800/50">
                  {activePrediction.mean_variance?.toExponential(2)}
                </span>
              </div>
            </div>
          ) : (
            /* ── MC Dropout still processing — show skeleton ── */
            <div className="text-white flex items-end justify-between pointer-events-auto">
              <div className="flex flex-col gap-1.5">
                <span className="text-[10px] text-gray-400 uppercase tracking-widest font-semibold">Uncertainty Estimate</span>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                  <span className="text-xs text-blue-300 font-medium animate-pulse">
                    MC Dropout processing…
                  </span>
                </div>
              </div>
              <div className="text-right flex flex-col gap-1.5">
                <span className="text-[10px] text-gray-400 uppercase tracking-widest font-semibold">MC Variance</span>
                <div className="h-5 w-20 bg-slate-700/60 rounded animate-pulse" />
              </div>
            </div>
          )}
        </div>
      )}

      {!activePrediction && (
        <div className="absolute bottom-0 inset-x-0 z-20 bg-gradient-to-t from-black/80 to-transparent p-4 flex justify-center pointer-events-none">
          <span className="text-xs text-gray-400 bg-black/50 px-3 py-1.5 rounded-full backdrop-blur-sm border border-white/10 pointer-events-auto">
            Select a finding from the summary panel to view its Grad-CAM overlay.
          </span>
        </div>
      )}
    </div>
  );
}
