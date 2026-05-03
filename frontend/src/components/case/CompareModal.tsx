'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import Image from 'next/image';
import { fetchCaseDetail } from '@/lib/api';
import type { CaseDetail } from '@/lib/types';
import {
  X, ZoomIn, ZoomOut, Maximize2, ArrowLeftRight, Loader2,
  AlertCircle, SlidersHorizontal, ChevronDown, ChevronUp,
} from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

interface CompareModalProps {
  currentCase: CaseDetail;
  similarCaseId: string;
  similarityScore?: number | null;
  onClose: () => void;
}

interface PanelState {
  brightness: number;
  contrast: number;
}

export function CompareModal({ currentCase, similarCaseId, similarityScore, onClose }: CompareModalProps) {
  const [similarCase, setSimilarCase] = useState<CaseDetail | null>(null);
  const [loading, setLoading]         = useState(true);
  const [loadError, setLoadError]     = useState('');
  const [swapped, setSwapped]         = useState(false);
  const [showControls, setShowControls] = useState(true);

  // Shared zoom / pan
  const [zoom,      setZoom]      = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });

  // Per-side enhancements
  const [leftPanel,  setLeft]  = useState<PanelState>({ brightness: 100, contrast: 100 });
  const [rightPanel, setRight] = useState<PanelState>({ brightness: 100, contrast: 100 });

  // Pan interaction
  const isPanning = useRef(false);
  const lastPos   = useRef({ x: 0, y: 0 });

  useEffect(() => {
    fetchCaseDetail(similarCaseId)
      .then(detail => { setSimilarCase(detail); setLoading(false); })
      .catch(err   => { setLoadError(err.message || 'Failed to load case'); setLoading(false); });
  }, [similarCaseId]);

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 's')      setSwapped(v => !v);
      if (e.key === '+' || e.key === '=') setZoom(z => Math.min(5, z + 0.2));
      if (e.key === '-')                  setZoom(z => Math.max(0.5, z - 0.2));
      if (e.key === '0')                  { setZoom(1); setPanOffset({ x: 0, y: 0 }); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const handlePointerDown = (e: React.PointerEvent) => {
    isPanning.current = true;
    lastPos.current = { x: e.clientX, y: e.clientY };
    if (e.target instanceof Element) e.target.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!isPanning.current) return;
    const dx = e.clientX - lastPos.current.x;
    const dy = e.clientY - lastPos.current.y;
    setPanOffset(prev => ({ x: prev.x + dx / zoom, y: prev.y + dy / zoom }));
    lastPos.current = { x: e.clientX, y: e.clientY };
  }, [zoom]);

  const handlePointerUp = (e: React.PointerEvent) => {
    isPanning.current = false;
    if (e.target instanceof Element && e.target.hasPointerCapture(e.pointerId)) {
      e.target.releasePointerCapture(e.pointerId);
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom(z => Math.max(0.5, Math.min(5, z + delta)));
  };

  // Determine which case goes on which side
  const leftCase  = swapped ? similarCase : currentCase;
  const rightCase = swapped ? currentCase : similarCase;
  const leftLabel  = swapped ? 'Similar Case' : 'Current Case';
  const rightLabel = swapped ? 'Current Case' : 'Similar Case';

  const getImageUrl = (c: CaseDetail | null) =>
    c?.case?.cxr_dicom_url || `/mock-data/dicoms/case_${c?.case?.id?.substring(0, 8) ?? 'default'}.png`;

  const getTopFinding = (c: CaseDetail | null) => {
    if (!c?.predictions?.length) return null;
    const top = [...c.predictions].filter(p => p.probability > 0).sort((a, b) => b.probability - a.probability)[0];
    return top ?? null;
  };

  const transformStyle = `scale(${zoom}) translate(${panOffset.x}px, ${panOffset.y}px)`;

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col bg-black/95"
      onWheel={handleWheel}
    >
      {/* ── Header bar ── */}
      <div className="flex items-center justify-between px-4 py-2 bg-slate-900 border-b border-slate-700 shrink-0">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-bold text-white">Case Comparison</h2>
          {similarityScore != null && (
            <span className="text-xs bg-blue-900/80 border border-blue-700 text-blue-200 px-2.5 py-0.5 rounded-full font-semibold">
              {Math.round(similarityScore)}% similarity
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Zoom controls */}
          <button onClick={() => setZoom(z => Math.max(0.5, z - 0.2))} className="p-1.5 bg-slate-700 hover:bg-slate-600 rounded text-white transition-colors" title="Zoom out (-)">
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <span className="text-xs text-slate-300 font-mono w-12 text-center">{zoom.toFixed(1)}x</span>
          <button onClick={() => setZoom(z => Math.min(5, z + 0.2))} className="p-1.5 bg-slate-700 hover:bg-slate-600 rounded text-white transition-colors" title="Zoom in (+)">
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => { setZoom(1); setPanOffset({ x: 0, y: 0 }); }} className="p-1.5 bg-slate-700 hover:bg-slate-600 rounded text-white transition-colors" title="Reset (0)">
            <Maximize2 className="w-3.5 h-3.5" />
          </button>

          <div className="w-px h-5 bg-slate-600 mx-1" />

          {/* Swap sides */}
          <button onClick={() => setSwapped(v => !v)} className="flex items-center gap-1.5 px-2.5 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-white text-xs font-medium transition-colors" title="Swap sides (S)">
            <ArrowLeftRight className="w-3.5 h-3.5" />
            Swap
          </button>

          {/* Toggle controls panel */}
          <button onClick={() => setShowControls(v => !v)} className="flex items-center gap-1 px-2.5 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-white text-xs font-medium transition-colors">
            <SlidersHorizontal className="w-3.5 h-3.5" />
            {showControls ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>

          <div className="w-px h-5 bg-slate-600 mx-1" />

          <button onClick={onClose} className="p-1.5 bg-slate-700 hover:bg-red-700/80 rounded text-white transition-colors" title="Close (Esc)">
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* ── Per-side controls ── */}
      {showControls && (
        <div className="flex items-stretch bg-slate-800 border-b border-slate-700 shrink-0">
          {[
            { side: 'left', cfg: leftPanel,  setCfg: setLeft,  label: leftLabel },
            { side: 'right', cfg: rightPanel, setCfg: setRight, label: rightLabel },
          ].map(({ side, cfg, setCfg, label }) => (
            <div key={side} className="flex-1 flex items-center gap-4 px-6 py-2 border-r border-slate-700 last:border-r-0">
              <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-widest shrink-0">{label}</span>
              <div className="flex items-center gap-3 flex-1">
                <label className="text-[10px] text-slate-400 shrink-0">Brightness</label>
                <input type="range" min="0" max="200" value={cfg.brightness}
                  onChange={(e) => setCfg(p => ({ ...p, brightness: +e.target.value }))}
                  className="flex-1 accent-blue-400 h-1" />
                <span className="text-[10px] font-mono text-slate-300 w-10 text-right">{cfg.brightness}%</span>
              </div>
              <div className="flex items-center gap-3 flex-1">
                <label className="text-[10px] text-slate-400 shrink-0">Contrast</label>
                <input type="range" min="0" max="200" value={cfg.contrast}
                  onChange={(e) => setCfg(p => ({ ...p, contrast: +e.target.value }))}
                  className="flex-1 accent-blue-400 h-1" />
                <span className="text-[10px] font-mono text-slate-300 w-10 text-right">{cfg.contrast}%</span>
              </div>
              <button onClick={() => setCfg({ brightness: 100, contrast: 100 })} className="text-[10px] text-slate-500 hover:text-slate-300 transition-colors shrink-0">Reset</button>
            </div>
          ))}
        </div>
      )}

      {/* ── Main comparison area ── */}
      <div className="flex-1 flex overflow-hidden">
        {loading ? (
          <div className="flex-1 flex items-center justify-center text-slate-400">
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-8 h-8 animate-spin" />
              <p className="text-sm">Loading similar case…</p>
            </div>
          </div>
        ) : loadError ? (
          <div className="flex-1 flex items-center justify-center text-red-400">
            <div className="flex flex-col items-center gap-3">
              <AlertCircle className="w-8 h-8" />
              <p className="text-sm">{loadError}</p>
            </div>
          </div>
        ) : (
          <>
            {/* Left panel */}
            <ComparePanel
              caseDetail={leftCase}
              label={leftLabel}
              panelState={leftPanel}
              transformStyle={transformStyle}
              isCurrentCase={!swapped}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              getImageUrl={getImageUrl}
              getTopFinding={getTopFinding}
            />

            {/* Divider */}
            <div className="w-[2px] bg-slate-600 shrink-0 relative">
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-slate-700 border border-slate-500 rounded-full p-1.5 text-slate-300">
                <ArrowLeftRight className="w-3 h-3" />
              </div>
            </div>

            {/* Right panel */}
            <ComparePanel
              caseDetail={rightCase}
              label={rightLabel}
              panelState={rightPanel}
              transformStyle={transformStyle}
              isCurrentCase={swapped}
              similarityScore={!swapped ? similarityScore : undefined}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              getImageUrl={getImageUrl}
              getTopFinding={getTopFinding}
            />
          </>
        )}
      </div>

      {/* ── Footer: keyboard shortcuts ── */}
      <div className="px-4 py-1.5 bg-slate-900 border-t border-slate-800 flex items-center gap-4 shrink-0">
        {[['S','Swap sides'],['+ / -','Zoom'],['0','Reset zoom'],['Drag','Pan'],['Esc','Close']].map(([key, desc]) => (
          <div key={key} className="flex items-center gap-1">
            <kbd className="text-[9px] bg-slate-700 text-slate-300 px-1.5 py-0.5 rounded font-mono">{key}</kbd>
            <span className="text-[10px] text-slate-500">{desc}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Inner panel component ────────────────────────────────────────────────────

interface ComparePanelProps {
  caseDetail: CaseDetail | null;
  label: string;
  panelState: PanelState;
  transformStyle: string;
  isCurrentCase: boolean;
  similarityScore?: number | null;
  onPointerDown: (e: React.PointerEvent) => void;
  onPointerMove: (e: React.PointerEvent) => void;
  onPointerUp:   (e: React.PointerEvent) => void;
  getImageUrl:   (c: CaseDetail | null) => string;
  getTopFinding: (c: CaseDetail | null) => any;
}

function ComparePanel({
  caseDetail, label, panelState, transformStyle, isCurrentCase,
  similarityScore, onPointerDown, onPointerMove, onPointerUp,
  getImageUrl, getTopFinding,
}: ComparePanelProps) {
  const imgUrl    = getImageUrl(caseDetail);
  const topFind   = getTopFinding(caseDetail);
  const patient   = caseDetail?.patient;
  const riskLevel = caseDetail?.case?.phase_a_risk_level;

  return (
    <div className="flex-1 flex flex-col bg-[#0a0f1e] overflow-hidden">
      {/* Panel header */}
      <div className="px-4 py-2 bg-slate-900/80 border-b border-slate-700/60 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <span className={cn(
            "text-[10px] font-bold px-2 py-0.5 rounded border uppercase tracking-widest",
            isCurrentCase ? "bg-blue-900/60 border-blue-700 text-blue-300" : "bg-slate-700/60 border-slate-600 text-slate-300"
          )}>
            {label}
          </span>
          {patient && (
            <span className="text-xs text-slate-300 font-medium">
              {patient.first_name} {patient.last_name} · {patient.mrn}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {similarityScore != null && (
            <span className="text-[10px] font-bold text-emerald-400 bg-emerald-900/40 border border-emerald-700/50 px-2 py-0.5 rounded-full">
              {Math.round(similarityScore)}% match
            </span>
          )}
          {riskLevel && (
            <span className={cn(
              "text-[10px] font-bold px-2 py-0.5 rounded border",
              riskLevel === 'High'     ? "bg-red-900/60 border-red-700 text-red-300" :
              riskLevel === 'Moderate' ? "bg-amber-900/60 border-amber-700 text-amber-300" :
                                         "bg-slate-700/60 border-slate-600 text-slate-300"
            )}>
              {riskLevel} Risk
            </span>
          )}
        </div>
      </div>

      {/* Image area — shared pan */}
      <div
        className="flex-1 relative overflow-hidden cursor-grab active:cursor-grabbing select-none touch-none"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <div
          className="absolute inset-0 origin-center transition-transform duration-75 ease-linear"
          style={{
            transform: transformStyle,
            filter: `brightness(${panelState.brightness}%) contrast(${panelState.contrast}%)`,
          }}
        >
          {imgUrl ? (
            <Image
              src={imgUrl}
              alt={`CXR: ${label}`}
              fill
              unoptimized
              draggable={false}
              className="object-contain pointer-events-none"
              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">
              No CXR available
            </div>
          )}
        </div>
      </div>

      {/* Panel footer: clinical summary */}
      <div className="px-4 py-3 bg-slate-900/90 border-t border-slate-700/60 grid grid-cols-3 gap-3 shrink-0 text-[11px]">
        <div>
          <p className="text-slate-500 uppercase tracking-widest font-semibold text-[9px] mb-0.5">Top Finding</p>
          <p className="text-white font-medium truncate">{topFind?.label ?? '—'}</p>
          {topFind && <p className="text-slate-400">{(topFind.probability * 100).toFixed(1)}% probability</p>}
        </div>
        <div>
          <p className="text-slate-500 uppercase tracking-widest font-semibold text-[9px] mb-0.5">Phase A Risk</p>
          <p className={cn("font-bold",
            riskLevel === 'High' ? 'text-red-400' : riskLevel === 'Moderate' ? 'text-amber-400' : 'text-slate-300'
          )}>{riskLevel ?? '—'}</p>
          {caseDetail?.case?.phase_a_risk_score != null && (
            <p className="text-slate-400">{(caseDetail.case.phase_a_risk_score * 100).toFixed(0)}% score</p>
          )}
        </div>
        <div>
          <p className="text-slate-500 uppercase tracking-widest font-semibold text-[9px] mb-0.5">Patient</p>
          <p className="text-white font-medium">{patient ? `${patient.sex}, ${patient.age_at_admission}y` : '—'}</p>
          <p className="text-slate-400">{caseDetail?.case?.admitted_at ? new Date(caseDetail.case.admitted_at).toLocaleDateString() : ''}</p>
        </div>
      </div>
    </div>
  );
}
