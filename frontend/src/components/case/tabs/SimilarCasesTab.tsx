'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useCaseStore } from '@/store/caseStore';
import { useToastStore } from '@/store/toastStore';
import { fetchSimilarCases } from '@/lib/api';
import { CaseSummary } from '@/lib/types';
import { CompareModal } from '@/components/case/CompareModal';
import {
  Search, SlidersHorizontal, ArrowRight, UserCheck, Stethoscope,
  GitCompare, CheckCircle2, Activity, AlertTriangle, RefreshCw, Clock,
  Layers,
} from 'lucide-react';
import Image from 'next/image';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

const SimilarityRing = ({ score }: { score: number }) => {
  const radius = 22;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  return (
    <div className="relative inline-flex items-center justify-center">
      <svg className="w-14 h-14 transform -rotate-90">
        <circle className="text-gray-100" strokeWidth="4" stroke="currentColor" fill="transparent" r={radius} cx="28" cy="28" />
        <circle
          className={cn("transition-all duration-1000 ease-out",
            score >= 90 ? "text-emerald-500" : score >= 80 ? "text-blue-500" : "text-amber-500"
          )}
          strokeWidth="4" strokeDasharray={circumference} strokeDashoffset={offset}
          strokeLinecap="round" stroke="currentColor" fill="transparent" r={radius} cx="28" cy="28"
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-sm font-bold text-gray-800">{score}%</span>
      </div>
    </div>
  );
};

export function SimilarCasesTab() {
  const currentCase = useCaseStore((state) => state.currentCase);
  const addToast = useToastStore((state) => state.addToast);

  const [isSearching, setIsSearching] = useState(true);
  const [similarCases, setSimilarCases] = useState<CaseSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [compareCase, setCompareCase] = useState<{ id: string; score?: number | null } | null>(null);
  const [topK, setTopK] = useState(3);
  const [lastSearched, setLastSearched] = useState<Date | null>(null);
  const [modality, setModality] = useState<'symile' | 'densenet' | 'unknown'>('unknown');
  const [indexSize, setIndexSize] = useState(0);

  const runSearch = useCallback((k: number, caseId: string) => {
    setIsSearching(true);
    setError(null);
    fetchSimilarCases(caseId, k)
      .then((data) => {
        setSimilarCases(data.cases);
        setModality(data.modality);
        setIndexSize(data.indexSize);
        setLastSearched(new Date());
        setIsSearching(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to retrieve similar cases.');
        setIsSearching(false);
      });
  }, []);

  useEffect(() => {
    if (!currentCase?.case.id) return;
    runSearch(topK, currentCase.case.id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentCase?.case.id]);

  const handleRerun = () => {
    if (!currentCase?.case.id) return;
    runSearch(topK, currentCase.case.id);
  };

  const handleTopKChange = (newK: number) => {
    setTopK(newK);
    if (currentCase?.case.id) runSearch(newK, currentCase.case.id);
  };

  if (!currentCase) {
    return <div className="h-full flex items-center justify-center text-gray-500">No case selected.</div>;
  }

  const patientName = currentCase.patient.first_name;

  return (
    <>
    <div className="w-full max-w-[1200px] mx-auto flex flex-col gap-6 py-4">

      {/* Search Header */}
      <div className="bg-white border border-gray-200 rounded-xl p-5 shadow-sm flex flex-col gap-4">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="bg-blue-50 p-3 rounded-xl border border-blue-100">
              {isSearching ? (
                <Search className="w-6 h-6 text-blue-600 animate-pulse" />
              ) : error ? (
                <AlertTriangle className="w-6 h-6 text-red-600" />
              ) : (
                <CheckCircle2 className="w-6 h-6 text-emerald-600" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <h2 className="text-lg font-bold text-gray-900">FAISS Dense Retrieval</h2>
                {!isSearching && !error && modality !== 'unknown' && (
                  <span
                    className={cn(
                      "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold border",
                      modality === 'symile'
                        ? "bg-violet-50 text-violet-700 border-violet-200"
                        : "bg-slate-50 text-slate-700 border-slate-200"
                    )}
                    title={modality === 'symile'
                      ? "Symile-MIMIC retrieval — query embedding fuses CXR + ECG + Labs"
                      : "DenseNet121 retrieval — query embedding uses CXR only"}
                  >
                    <Layers className="w-3 h-3" />
                    {modality === 'symile' ? 'Multimodal (CXR + ECG + Labs)' : 'CXR-only'}
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-500">
                {isSearching
                  ? `Querying multimodal embedding space for ${patientName}…`
                  : error
                  ? `Search unavailable: ${error}`
                  : `Found ${similarCases.length} similar historical cohorts for ${patientName}.`}
              </p>
            </div>
          </div>

          {/* Controls row */}
          <div className="flex items-center gap-2 flex-wrap">
            {/* Top-K selector */}
            <div className="flex items-center bg-gray-100 rounded-lg p-0.5 gap-0.5">
              {[3, 5, 10].map(k => (
                <button
                  key={k}
                  onClick={() => handleTopKChange(k)}
                  className={cn(
                    "px-3 py-1.5 rounded-md text-xs font-semibold transition-all",
                    topK === k ? "bg-white text-gray-800 shadow-sm" : "text-gray-500 hover:text-gray-700"
                  )}
                >
                  Top {k}
                </button>
              ))}
            </div>

            {/* Rerun button */}
            <button
              onClick={handleRerun}
              disabled={isSearching}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white rounded-lg text-sm font-medium transition-colors shadow-sm whitespace-nowrap"
            >
              <RefreshCw className={cn("w-4 h-4", isSearching && "animate-spin")} />
              {isSearching ? 'Searching…' : 'Rerun FAISS'}
            </button>
          </div>
        </div>

        {/* Timestamp */}
        {lastSearched && !isSearching && (
          <div className="flex items-center gap-1.5 text-[11px] text-gray-400 border-t border-gray-100 pt-2 flex-wrap">
            <Clock className="w-3 h-3" />
            Last searched: {lastSearched.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            <span className="mx-1 text-gray-200">·</span>
            Showing top {topK} results
            {indexSize > 0 && (
              <>
                <span className="mx-1 text-gray-200">·</span>
                Index: {indexSize} {modality === 'symile' ? 'multimodal' : 'CXR'} embeddings
              </>
            )}
          </div>
        )}
      </div>

      {/* Loading Skeleton */}
      {isSearching && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {Array.from({ length: topK > 5 ? 6 : topK }).map(i => (
            <div key={String(i)} className="bg-white border border-gray-100 rounded-2xl p-5 shadow-sm animate-pulse h-[300px]">
              <div className="h-40 bg-gray-200 rounded-xl mb-4 w-full" />
              <div className="h-6 bg-gray-200 rounded w-3/4 mb-2" />
              <div className="h-4 bg-gray-200 rounded w-1/2" />
            </div>
          ))}
        </div>
      )}

      {/* Error */}
      {!isSearching && error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center text-red-700">
          <p className="font-medium">Retrieval Failed</p>
          <p className="text-sm mt-1">{error}</p>
          <button
            onClick={handleRerun}
            className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {/* Empty */}
      {!isSearching && !error && similarCases.length === 0 && (
        <div className="bg-gray-50 border border-gray-200 rounded-xl p-8 text-center">
          <p className="text-gray-500 font-medium">No similar cases found in the index.</p>
          <p className="text-xs text-gray-400 mt-1">Run inference first to generate the CXR embedding for this case.</p>
        </div>
      )}

      {/* Precedent Cards Grid */}
      {!isSearching && !error && similarCases.length > 0 && (
        <div className={cn(
          "grid gap-6",
          similarCases.length <= 3 ? "grid-cols-1 lg:grid-cols-3" : "grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
        )}>
          {similarCases.map((prec, rank) => {
            const similarity = prec.similarity_score != null ? Math.round(prec.similarity_score) : null;
            return (
              <div
                key={prec.case_id}
                className="bg-white border border-gray-100 rounded-2xl overflow-hidden shadow-card hover:shadow-card-hover hover:-translate-y-1 hover:border-gray-200 transition-all duration-200 group flex flex-col opacity-0 animate-fadeInUp"
                style={{ animationDelay: `${rank * 70}ms`, animationFillMode: 'both' }}
              >

                {/* Rank badge */}
                <div className="relative h-48 bg-slate-900 overflow-hidden">
                  <Image
                    src={prec.cxr_dicom_url || `/mock-data/dicoms/case_${prec.case_id.substring(0, 8)}.png`}
                    alt={`CXR for ${prec.mrn}`}
                    fill
                    className="object-contain opacity-80 group-hover:opacity-100 transition-opacity duration-300"
                    unoptimized
                    onError={(e) => {
                      const img = e.target as HTMLImageElement;
                      if (!img.dataset.fallback) {
                        img.dataset.fallback = '1';
                        img.src = `/mock-data/dicoms/case_${prec.case_id.substring(0, 8)}.png`;
                      }
                    }}
                  />
                  <div className="absolute top-4 left-4 bg-black/60 backdrop-blur-md px-2.5 py-1 rounded text-xs font-mono text-white border border-white/20">
                    {prec.mrn}
                  </div>
                  <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-black/50 backdrop-blur-sm px-2 py-0.5 rounded text-[9px] font-bold text-white/60 uppercase tracking-widest">
                    #{rank + 1}
                  </div>
                  {similarity != null && (
                    <div className="absolute top-3 right-3 bg-white rounded-full shadow-lg">
                      <SimilarityRing score={similarity} />
                    </div>
                  )}
                  <div className="absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-white to-transparent" />
                </div>

                {/* Content */}
                <div className="p-5 flex-1 flex flex-col bg-white">
                  <h3 className="font-bold text-gray-900 text-lg mb-4 flex items-center gap-2">
                    <UserCheck className="w-5 h-5 text-gray-400" />
                    {prec.patient_name}
                  </h3>
                  <div className="space-y-3 flex-1">
                    <div>
                      <span className="text-[10px] uppercase font-bold tracking-widest text-gray-400 flex items-center gap-1.5 mb-1">
                        <Stethoscope className="w-3 h-3" /> Documented Findings
                      </span>
                      {prec.ground_truth_findings && prec.ground_truth_findings.length > 0 ? (
                        <div className="flex flex-wrap gap-1">
                          {prec.ground_truth_findings.map((label) => (
                            <span
                              key={label}
                              className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium bg-blue-50 text-blue-700 border border-blue-200"
                            >
                              {label}
                            </span>
                          ))}
                        </div>
                      ) : (
                        <p className="text-sm font-medium text-gray-800">
                          {prec.top_finding_label || 'No documented findings'}
                        </p>
                      )}
                    </div>
                    <div>
                      <span className="text-[10px] uppercase font-bold tracking-widest text-gray-400 flex items-center gap-1.5 mb-1">
                        <Activity className="w-3 h-3" /> Risk Level
                      </span>
                      <p className="text-sm text-gray-600 leading-relaxed">{prec.phase_a_risk_level || 'Monitor'}</p>
                    </div>
                  </div>

                  <div className="mt-5 pt-4 border-t border-gray-100">
                    <div className="flex justify-between items-center mb-4">
                      <span className="text-[10px] uppercase font-bold tracking-widest text-gray-400">Clinical Outcome</span>
                      <span className={cn(
                        "text-xs font-semibold px-2 py-0.5 rounded border",
                        prec.phase_a_risk_level === 'High' ? "bg-red-50 text-red-700 border-red-200" : "bg-emerald-50 text-emerald-700 border-emerald-200"
                      )}>
                        {prec.phase_a_risk_level === 'High' ? 'ICU Admission' : 'General Ward'}
                      </span>
                    </div>
                    <button
                      onClick={() => setCompareCase({ id: prec.case_id, score: prec.similarity_score })}
                      className="group/btn w-full flex items-center justify-center gap-2 py-2.5 bg-blue-50 hover:bg-blue-600 text-blue-700 hover:text-white rounded-lg text-sm font-semibold transition-all duration-150 border border-blue-200 hover:border-transparent hover:shadow-sm"
                    >
                      <GitCompare className="w-4 h-4" />
                      Compare with {patientName}
                      <ArrowRight className="w-3.5 h-3.5 opacity-0 group-hover/btn:opacity-100 group-hover/btn:translate-x-0.5 transition-all" />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

    </div>

    {compareCase && currentCase && (
      <CompareModal
        currentCase={currentCase}
        similarCaseId={compareCase.id}
        similarityScore={compareCase.score}
        onClose={() => setCompareCase(null)}
      />
    )}
    </>
  );
}
