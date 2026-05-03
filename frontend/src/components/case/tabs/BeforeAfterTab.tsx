'use client';

import React, { useState } from 'react';
import { useCaseStore } from '@/store/caseStore';
import Image from 'next/image';
import {
  Activity, Beaker, BrainCircuit, Clock, ArrowRight, CheckCircle2,
  AlertTriangle, Layers, TrendingUp, TrendingDown, Minus, Eye, EyeOff,
  Stethoscope, BadgeCheck,
} from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

function DataRow({ label, value, unit, highlight }: { label: string; value: string | number | null | undefined; unit?: string; highlight?: 'high' | 'low' | 'normal' }) {
  if (value == null) return null;
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-white/10 last:border-0">
      <span className="text-[11px] text-slate-400 font-medium">{label}</span>
      <span className={cn(
        "text-[12px] font-bold font-mono",
        highlight === 'high' ? 'text-red-300' : highlight === 'low' ? 'text-amber-300' : 'text-white'
      )}>
        {value}{unit ? ` ${unit}` : ''}
      </span>
    </div>
  );
}

function RiskPill({ level }: { level: string | null | undefined }) {
  if (!level) return <span className="text-gray-400 text-sm">N/A</span>;
  return (
    <span className={cn(
      "px-3 py-1.5 rounded-full text-sm font-bold border",
      level === 'High' ? 'bg-red-900/60 border-red-500 text-red-200' :
      level === 'Moderate' ? 'bg-amber-900/60 border-amber-500 text-amber-200' :
      'bg-emerald-900/60 border-emerald-500 text-emerald-200'
    )}>
      {level} Risk
    </span>
  );
}

export function BeforeAfterTab() {
  const currentCase = useCaseStore((state) => state.currentCase);
  const [showGradCam, setShowGradCam] = useState(true);

  if (!currentCase) {
    return <div className="h-full flex items-center justify-center text-gray-500">No case selected.</div>;
  }

  const { case: caseData, patient, predictions } = currentCase;
  const { ecg_data, lab_data } = caseData;

  const sortedPreds = [...predictions]
    .filter(p => p.probability > 0)
    .sort((a, b) => b.probability - a.probability);

  const topPred = sortedPreds[0];
  const topGradCam = topPred?.gradcam_url;
  const hasImage = !!caseData.cxr_dicom_url;

  // Agreement logic: compare Phase A risk with Phase B top finding severity
  const phaseAHighRisk = caseData.phase_a_risk_level === 'High' || caseData.phase_a_risk_level === 'Moderate';
  const phaseBHighRisk = topPred && topPred.probability > 0.6 && topPred.risk_badge === 'Elevated Risk';
  const hasPhaseBData = sortedPreds.length > 0;

  const agreement = !hasPhaseBData
    ? 'pending'
    : phaseAHighRisk === phaseBHighRisk
    ? 'consistent'
    : 'divergent';

  const admittedAt = caseData.admitted_at ? new Date(caseData.admitted_at).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }) : 'N/A';

  return (
    <div className="w-full max-w-[1400px] mx-auto flex flex-col gap-6 py-4">

      {/* Timeline header */}
      <div className="flex items-center justify-between bg-white border border-gray-200 rounded-xl px-5 py-4 shadow-sm">
        <div>
          <h2 className="text-lg font-bold text-gray-900">Clinical Timeline — {patient.first_name} {patient.last_name}</h2>
          <p className="text-sm text-gray-500">Admission: {admittedAt} · MRN: {patient.mrn} · {patient.sex}, {patient.age_at_admission}y</p>
        </div>

        {/* Agreement indicator */}
        <div className={cn(
          "flex items-center gap-2 px-4 py-2.5 rounded-xl border font-semibold text-sm",
          agreement === 'consistent' ? 'bg-emerald-50 border-emerald-200 text-emerald-700' :
          agreement === 'divergent'  ? 'bg-red-50 border-red-200 text-red-700' :
          'bg-gray-50 border-gray-200 text-gray-500'
        )}>
          {agreement === 'consistent' && <><CheckCircle2 className="w-4 h-4" />Phase A/B Consistent</>}
          {agreement === 'divergent'  && <><AlertTriangle className="w-4 h-4" />Phase A/B Divergent</>}
          {agreement === 'pending'    && <><Clock className="w-4 h-4" />Awaiting Phase B</>}
        </div>
      </div>

      {/* Prominent divergence banner — fires only when Phase A and Phase B
          disagree. Designed to be impossible to miss; explains *what* diverged
          and recommends correlation. */}
      {agreement === 'divergent' && (
        <div className="relative overflow-hidden rounded-2xl border-2 border-red-300 bg-gradient-to-r from-red-50 via-rose-50 to-red-50 shadow-md animate-fadeInDown">
          <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-gradient-to-b from-red-500 to-rose-600" />
          <div className="absolute -left-3 -top-3 w-20 h-20 bg-red-500/10 rounded-full blur-xl pointer-events-none" />
          <div className="flex items-start gap-4 px-6 py-5">
            <div className="shrink-0 w-12 h-12 rounded-2xl bg-red-100 border border-red-200 flex items-center justify-center">
              <AlertTriangle className="w-6 h-6 text-red-600 animate-pulseSoft" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] font-bold uppercase tracking-widest text-red-700 bg-red-100 border border-red-200 px-2 py-0.5 rounded-full">
                  Divergence Alert
                </span>
                <span className="text-[10px] font-mono text-red-500/70">CLINICAL CORRELATION RECOMMENDED</span>
              </div>
              <h3 className="text-base font-bold text-red-900 leading-snug">
                Phase A and Phase B risk assessments disagree
              </h3>
              <p className="text-sm text-red-800/80 mt-1 leading-relaxed">
                {phaseAHighRisk
                  ? <>ECG / lab profile suggests <strong>elevated risk</strong>, but the CXR top finding ({topPred?.label}) is <strong>not flagged as Elevated Risk</strong>. The cardiac/laboratory concern is not visibly explained by the imaging — consider non-imaging-evident pathology (early ischemia, sepsis, metabolic) before disposition.</>
                  : <>CXR shows <strong>{topPred?.label}</strong> at {topPred ? (topPred.probability * 100).toFixed(0) : '0'}% probability, but the ECG / lab profile is <strong>{caseData.phase_a_risk_level || 'low'}</strong>. The imaging finding is not corroborated by tabular signals — consider whether the imaging severity warrants admission independent of lab results.</>
                }
              </p>
              <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white border border-red-100">
                  <span className="w-2 h-2 rounded-full bg-blue-500 shrink-0" />
                  <span className="font-semibold text-slate-700">Phase A:</span>
                  <span className="text-slate-600">{caseData.phase_a_risk_level || 'N/A'} risk</span>
                  {caseData.phase_a_risk_score != null && (
                    <span className="ml-auto font-mono text-slate-400">{(caseData.phase_a_risk_score * 100).toFixed(0)}%</span>
                  )}
                </div>
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white border border-red-100">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0" />
                  <span className="font-semibold text-slate-700">Phase B:</span>
                  <span className="text-slate-600 truncate">{topPred?.label || 'N/A'}</span>
                  {topPred && (
                    <span className="ml-auto font-mono text-slate-400">{(topPred.probability * 100).toFixed(0)}%</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Two-column layout with timeline connector */}
      <div className="relative grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Timeline connector (visible on lg+) */}
        <div className="hidden lg:flex absolute inset-y-0 left-1/2 -translate-x-1/2 z-10 items-center pointer-events-none">
          <div className="w-8 h-8 rounded-full bg-white border-2 border-blue-500 flex items-center justify-center shadow-md">
            <ArrowRight className="w-4 h-4 text-blue-600" />
          </div>
        </div>

        {/* ─── PHASE A: Before Imaging ─────────────────────────────────────── */}
        <div className="flex flex-col rounded-2xl overflow-hidden border border-blue-200 shadow-card animate-fadeInUp">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-blue-700 px-5 py-4 flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <div className="bg-white/20 rounded-full px-2 py-0.5 text-[10px] font-bold text-white uppercase tracking-widest">Phase A</div>
              </div>
              <h3 className="text-white font-bold text-lg">Before Imaging</h3>
              <p className="text-blue-200 text-xs">Tabular ECG + Lab risk stratification</p>
            </div>
            <div className="flex flex-col items-end gap-1">
              <RiskPill level={caseData.phase_a_risk_level} />
              {caseData.phase_a_risk_score != null && (
                <span className="text-[10px] text-blue-200 font-mono">{(caseData.phase_a_risk_score * 100).toFixed(1)}% score</span>
              )}
            </div>
          </div>

          <div className="bg-[#0f172a] flex-1 p-5 space-y-5">

            {/* ECG */}
            {ecg_data && (
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Activity className="w-4 h-4 text-blue-400" />
                  <span className="text-xs font-bold text-blue-300 uppercase tracking-widest">Electrocardiogram</span>
                  {ecg_data.acquired_at && (
                    <span className="ml-auto text-[10px] text-slate-500 font-mono">{new Date(ecg_data.acquired_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  )}
                </div>
                <div className="bg-blue-900/20 border border-blue-800/40 rounded-lg p-3">
                  <div className="text-[10px] text-blue-400 font-medium mb-2">
                    {ecg_data.rhythm_interpretation || 'Unknown rhythm'}
                  </div>
                  <DataRow label="Heart Rate" value={ecg_data.heart_rate} unit="bpm"
                    highlight={ecg_data.heart_rate > 100 ? 'high' : ecg_data.heart_rate < 60 ? 'low' : 'normal'} />
                  <DataRow label="PR Interval" value={ecg_data.pr_interval_ms} unit="ms"
                    highlight={ecg_data.pr_interval_ms > 200 ? 'high' : undefined} />
                  <DataRow label="QRS Duration" value={ecg_data.qrs_duration_ms} unit="ms"
                    highlight={ecg_data.qrs_duration_ms > 120 ? 'high' : undefined} />
                  <DataRow label="QTc" value={ecg_data.qtc_ms} unit="ms"
                    highlight={ecg_data.qtc_ms > 450 ? 'high' : undefined} />
                  <DataRow label="ST Deviation" value={ecg_data.st_deviation_mm != null ? (ecg_data.st_deviation_mm > 0 ? `+${ecg_data.st_deviation_mm}` : ecg_data.st_deviation_mm) : null} unit="mm"
                    highlight={Math.abs(ecg_data.st_deviation_mm ?? 0) > 1 ? 'high' : 'normal'} />
                </div>
              </div>
            )}

            {/* Labs */}
            {lab_data && (
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Beaker className="w-4 h-4 text-purple-400" />
                  <span className="text-xs font-bold text-purple-300 uppercase tracking-widest">Laboratory Results</span>
                  {lab_data.collected_at && (
                    <span className="ml-auto text-[10px] text-slate-500 font-mono">{new Date(lab_data.collected_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                  )}
                </div>
                <div className="bg-purple-900/20 border border-purple-800/40 rounded-lg p-3">
                  <DataRow label="Troponin" value={lab_data.troponin_ng_ml} unit="ng/mL"
                    highlight={lab_data.troponin_ng_ml > 0.04 ? 'high' : 'normal'} />
                  <DataRow label="BNP" value={lab_data.bnp_pg_ml} unit="pg/mL"
                    highlight={lab_data.bnp_pg_ml > 100 ? 'high' : 'normal'} />
                  <DataRow label="WBC" value={lab_data.wbc_count} unit="×10³/μL"
                    highlight={lab_data.wbc_count > 11 ? 'high' : lab_data.wbc_count < 4.5 ? 'low' : 'normal'} />
                  <DataRow label="Creatinine" value={lab_data.creatinine_mg_dl} unit="mg/dL"
                    highlight={lab_data.creatinine_mg_dl > 1.3 ? 'high' : 'normal'} />
                  <DataRow label="Sodium" value={lab_data.sodium_meq_l} unit="mEq/L"
                    highlight={lab_data.sodium_meq_l > 145 ? 'high' : lab_data.sodium_meq_l < 136 ? 'low' : 'normal'} />
                  <DataRow label="Potassium" value={lab_data.potassium_meq_l} unit="mEq/L"
                    highlight={lab_data.potassium_meq_l > 5.0 ? 'high' : lab_data.potassium_meq_l < 3.5 ? 'low' : 'normal'} />
                  <DataRow label="Lactate" value={lab_data.lactate_mmol_l} unit="mmol/L"
                    highlight={lab_data.lactate_mmol_l > 2.0 ? 'high' : 'normal'} />
                </div>
              </div>
            )}

            {/* Recommendation */}
            {caseData.phase_a_recommendation && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Stethoscope className="w-4 h-4 text-slate-400" />
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">Clinical Recommendation</span>
                </div>
                <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg px-4 py-3">
                  <p className="text-sm text-slate-300 leading-relaxed">{caseData.phase_a_recommendation}</p>
                </div>
              </div>
            )}

            {!ecg_data && !lab_data && (
              <div className="text-center py-8 text-slate-500 text-sm">No Phase A data available.</div>
            )}
          </div>
        </div>

        {/* ─── PHASE B: After Imaging ──────────────────────────────────────── */}
        <div className="flex flex-col rounded-2xl overflow-hidden border border-emerald-200 shadow-card animate-fadeInUp stagger-1">
          {/* Header */}
          <div className="bg-gradient-to-r from-emerald-600 to-teal-700 px-5 py-4 flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <div className="bg-white/20 rounded-full px-2 py-0.5 text-[10px] font-bold text-white uppercase tracking-widest">Phase B</div>
              </div>
              <h3 className="text-white font-bold text-lg">After Imaging</h3>
              <p className="text-emerald-200 text-xs">CXR + Grad-CAM + DenseNet121 predictions</p>
            </div>
            <div className="flex flex-col items-end gap-2">
              {hasPhaseBData && topPred ? (
                <>
                  <span className={cn(
                    "px-3 py-1.5 rounded-full text-sm font-bold border",
                    topPred.risk_badge === 'Elevated Risk' ? 'bg-red-900/60 border-red-500 text-red-200' :
                    topPred.risk_badge === 'Monitor' ? 'bg-amber-900/60 border-amber-500 text-amber-200' :
                    'bg-emerald-900/60 border-emerald-500 text-emerald-200'
                  )}>
                    {topPred.label}
                  </span>
                  {topPred.probability >= 0.85 && (
                    <span className="flex items-center gap-1 text-[10px] font-bold text-emerald-300 bg-emerald-900/40 border border-emerald-700/50 px-2 py-0.5 rounded-full">
                      <BadgeCheck className="w-3 h-3" />Verified TP
                    </span>
                  )}
                </>
              ) : (
                <span className="text-emerald-200 text-xs">Awaiting inference</span>
              )}
            </div>
          </div>

          <div className="bg-[#0f172a] flex-1 flex flex-col p-5 space-y-4">

            {/* CXR Image */}
            <div className="relative rounded-xl overflow-hidden bg-black border border-slate-700" style={{ height: '260px' }}>
              {hasImage ? (
                <>
                  <Image
                    src={caseData.cxr_dicom_url!}
                    alt="CXR"
                    fill
                    unoptimized
                    className="object-contain"
                  />
                  {showGradCam && topGradCam && (
                    <Image
                      src={topGradCam}
                      alt="Grad-CAM overlay"
                      fill
                      unoptimized
                      className="object-contain"
                      style={{ mixBlendMode: 'multiply' }}
                    />
                  )}
                  {/* Toggle overlay button */}
                  <button
                    onClick={() => setShowGradCam(v => !v)}
                    className="absolute bottom-2 right-2 flex items-center gap-1 bg-black/70 text-white text-[10px] font-semibold px-2 py-1 rounded backdrop-blur-sm border border-white/10 hover:bg-white/20 transition-colors"
                  >
                    {showGradCam ? <><EyeOff className="w-3 h-3" /> Hide Grad-CAM</> : <><Eye className="w-3 h-3" /> Show Grad-CAM</>}
                  </button>
                  {showGradCam && topGradCam && (
                    <div className="absolute top-2 left-2 bg-black/60 text-emerald-300 text-[9px] font-bold px-2 py-1 rounded backdrop-blur-sm border border-emerald-800/50">
                      GRAD-CAM: {topPred?.label}
                    </div>
                  )}
                  {!topGradCam && (
                    <div className="absolute top-2 left-2 bg-black/60 text-slate-400 text-[9px] font-bold px-2 py-1 rounded backdrop-blur-sm border border-slate-700/50">
                      NO GRAD-CAM — RUN INFERENCE
                    </div>
                  )}
                </>
              ) : (
                <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-2">
                  <Layers className="w-8 h-8 opacity-40" />
                  <p className="text-sm font-semibold text-slate-400">No CXR available</p>
                  <p className="text-[11px] text-slate-500">Upload via the modality indicators above</p>
                </div>
              )}
            </div>

            {/* Predictions */}
            <div>
              <div className="flex items-center gap-2 mb-3">
                <BrainCircuit className="w-4 h-4 text-emerald-400" />
                <span className="text-xs font-bold text-emerald-300 uppercase tracking-widest">Model Predictions</span>
                {topPred && <span className="ml-auto text-[10px] text-slate-500">{predictions.filter(p => p.probability > 0).length} findings</span>}
              </div>

              {sortedPreds.length === 0 ? (
                <div className="bg-slate-800/40 border border-slate-700/40 rounded-lg px-4 py-6 text-center text-slate-500 text-xs">
                  Run inference to generate predictions
                </div>
              ) : (
                <div className="space-y-2">
                  {sortedPreds.slice(0, 5).map(pred => (
                    <div key={pred.id} className="flex items-center gap-3 bg-slate-800/40 border border-slate-700/30 rounded-lg px-3 py-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5 mb-1">
                          <span className="text-[11px] font-semibold text-white truncate">{pred.label}</span>
                          {pred.probability >= 0.85 && (
                            <BadgeCheck className="w-3 h-3 text-emerald-400 shrink-0" />
                          )}
                        </div>
                        <div className="w-full h-1 bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className={cn(
                              "h-full rounded-full",
                              pred.probability > 0.7 ? 'bg-red-400' : pred.probability > 0.5 ? 'bg-amber-400' : 'bg-emerald-400'
                            )}
                            style={{ width: `${pred.probability * 100}%` }}
                          />
                        </div>
                      </div>
                      <span className="text-[11px] font-bold font-mono text-white shrink-0">{(pred.probability * 100).toFixed(1)}%</span>
                      {pred.uncertainty_level && (
                        <span className={cn(
                          "text-[9px] font-bold px-1.5 py-0.5 rounded border shrink-0",
                          pred.uncertainty_level === 'High Uncertainty' ? 'bg-red-900/50 border-red-700 text-red-300' :
                          pred.uncertainty_level === 'Moderate Uncertainty' ? 'bg-amber-900/50 border-amber-700 text-amber-300' :
                          'bg-emerald-900/50 border-emerald-700 text-emerald-300'
                        )}>
                          {pred.uncertainty_level.replace(' Uncertainty', '')}
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ─── Comparison Summary ──────────────────────────────────────────────── */}
      {hasPhaseBData && (
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
          <div className="bg-slate-50 border-b border-gray-200 px-5 py-3 flex items-center gap-2">
            <ArrowRight className="w-4 h-4 text-slate-500" />
            <h3 className="font-bold text-slate-800 text-sm">Phase A → Phase B Comparison</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x divide-gray-100">
            {/* Risk evolution */}
            <div className="p-5 flex flex-col gap-2">
              <span className="text-[10px] uppercase font-bold tracking-widest text-gray-400">Risk Evolution</span>
              <div className="flex items-center gap-3">
                <span className={cn(
                  "text-sm font-bold px-2 py-1 rounded-lg",
                  caseData.phase_a_risk_level === 'High' ? 'bg-red-100 text-red-700' :
                  caseData.phase_a_risk_level === 'Moderate' ? 'bg-amber-100 text-amber-700' :
                  'bg-emerald-100 text-emerald-700'
                )}>
                  {caseData.phase_a_risk_level || 'N/A'}
                </span>
                <ArrowRight className="w-4 h-4 text-gray-300" />
                <span className={cn(
                  "text-sm font-bold px-2 py-1 rounded-lg",
                  topPred?.risk_badge === 'Elevated Risk' ? 'bg-red-100 text-red-700' :
                  topPred?.risk_badge === 'Monitor' ? 'bg-amber-100 text-amber-700' :
                  'bg-emerald-100 text-emerald-700'
                )}>
                  {topPred?.risk_badge || 'N/A'}
                </span>
                {caseData.phase_a_risk_level === 'High' && topPred?.risk_badge === 'Elevated Risk' && <TrendingUp className="w-4 h-4 text-red-500" />}
                {caseData.phase_a_risk_level === 'Low' && topPred?.risk_badge === 'Unlikely' && <TrendingDown className="w-4 h-4 text-emerald-500" />}
                {agreement === 'consistent' && <Minus className="w-4 h-4 text-gray-400" />}
              </div>
            </div>

            {/* Top finding */}
            <div className="p-5 flex flex-col gap-2">
              <span className="text-[10px] uppercase font-bold tracking-widest text-gray-400">Top CXR Finding</span>
              {topPred ? (
                <div>
                  <p className="text-sm font-bold text-gray-800">{topPred.label}</p>
                  <p className="text-xs text-gray-500">{(topPred.probability * 100).toFixed(1)}% confidence</p>
                  {topPred.probability >= 0.85 && (
                    <p className="text-[10px] text-emerald-600 font-semibold flex items-center gap-1 mt-1">
                      <BadgeCheck className="w-3 h-3" />High-confidence — likely true positive
                    </p>
                  )}
                </div>
              ) : <p className="text-sm text-gray-400">No predictions yet</p>}
            </div>

            {/* Agreement */}
            <div className="p-5 flex flex-col gap-2">
              <span className="text-[10px] uppercase font-bold tracking-widest text-gray-400">Phase Agreement</span>
              <div className={cn(
                "flex items-center gap-2 text-sm font-bold",
                agreement === 'consistent' ? 'text-emerald-600' :
                agreement === 'divergent'  ? 'text-red-600' : 'text-gray-400'
              )}>
                {agreement === 'consistent' && <><CheckCircle2 className="w-5 h-5" />Consistent findings</>}
                {agreement === 'divergent'  && <><AlertTriangle className="w-5 h-5" />Divergent — review recommended</>}
                {agreement === 'pending'    && <>Awaiting Phase B data</>}
              </div>
              {agreement === 'divergent' && (
                <p className="text-xs text-red-500">Phase A and Phase B risk assessments differ — clinical correlation advised.</p>
              )}
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
