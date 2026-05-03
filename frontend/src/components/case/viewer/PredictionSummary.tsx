'use client';

import React from 'react';
import { useCaseStore } from '@/store/caseStore';
import { ChevronRight, Filter, BadgeCheck } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

const UNCERTAINTY_BADGE: Record<string, { bg: string; text: string; border: string; label: string }> = {
  'Low Uncertainty':      { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200', label: 'Low Uncertainty' },
  'Moderate Uncertainty': { bg: 'bg-amber-50',   text: 'text-amber-700',   border: 'border-amber-300',   label: 'Moderate Uncertainty' },
  'High Uncertainty':     { bg: 'bg-red-50',     text: 'text-red-700',     border: 'border-red-400',     label: 'High Uncertainty' },
};

export function PredictionSummary() {
  const currentCase = useCaseStore((state) => state.currentCase);
  const selectedLabel = useCaseStore((state) => state.selectedLabel);
  const setSelectedLabel = useCaseStore((state) => state.setSelectedLabel);
  const threshold = useCaseStore((state) => state.probabilityThreshold);
  const setThreshold = useCaseStore((state) => state.setProbabilityThreshold);

  if (!currentCase) return null;

  const { predictions } = currentCase;
  // Filter out unmodeled labels (prob=0) that come from the 8→14 mapping
  const modeledPredictions = predictions.filter(p => p.probability > 0);
  const sortedPredictions = [...modeledPredictions].sort((a, b) => b.probability - a.probability);
  const aboveThreshold = sortedPredictions.filter(p => p.probability >= threshold);
  const belowThreshold = sortedPredictions.filter(p => p.probability < threshold);

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm h-full flex flex-col overflow-hidden">
      <div className="bg-gray-50 border-b border-gray-200 px-4 py-3">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-gray-900 text-sm">Model Predictions</h3>
          <span className="text-xs text-gray-500 font-medium bg-gray-200 px-2 py-0.5 rounded-full">
            {aboveThreshold.length}/{predictions.length}
          </span>
        </div>
        {/* Threshold Slider */}
        <div className="flex items-center gap-2">
          <Filter className="w-3.5 h-3.5 text-gray-400 shrink-0" />
          <input
            type="range" min="0" max="100" step="5"
            value={Math.round(threshold * 100)}
            onChange={(e) => setThreshold(parseInt(e.target.value) / 100)}
            className="flex-1 accent-blue-600 h-1"
          />
          <span className="text-[10px] font-mono text-gray-500 w-8 text-right">
            {Math.round(threshold * 100)}%
          </span>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {aboveThreshold.map((pred) => {
          const uncertaintyStyle = pred.uncertainty_level
            ? UNCERTAINTY_BADGE[pred.uncertainty_level] ?? UNCERTAINTY_BADGE['Low Uncertainty']
            : null;
          const isSelected = selectedLabel === pred.label;
          const probabilityPct = (pred.probability * 100).toFixed(1);
          const isVerifiedTP = pred.probability >= 0.85;

          return (
            <button
              key={pred.id}
              onClick={() => setSelectedLabel(pred.label)}
              className={cn(
                "w-full text-left px-3 py-3 rounded-lg border transition-all duration-200 group flex flex-col gap-3",
                isSelected
                  ? "bg-blue-50 border-blue-300 ring-1 ring-blue-500 shadow-sm"
                  : "bg-white border-gray-100 hover:border-gray-300 hover:bg-gray-50"
              )}
            >
              <div className="flex items-center justify-between w-full">
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className={cn(
                    "font-semibold text-sm truncate",
                    isSelected ? "text-blue-900" : "text-gray-800"
                  )}>
                    {pred.label}
                  </span>
                  {isVerifiedTP && (
                    <span className="inline-flex items-center gap-0.5 text-[9px] font-bold text-emerald-700 bg-emerald-50 border border-emerald-200 px-1.5 py-0.5 rounded-full shrink-0" title="Model confidence ≥85% — likely verified true positive">
                      <BadgeCheck className="w-2.5 h-2.5" />
                      TP
                    </span>
                  )}
                </div>
                <span className="font-bold text-sm text-gray-900 tabular-nums">
                  {probabilityPct}%
                </span>
              </div>
              
              {/* Probability bar */}
              <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={cn(
                    "h-full rounded-full transition-all",
                    pred.probability > 0.7 ? "bg-red-400" :
                    pred.probability > 0.5 ? "bg-amber-400" : "bg-emerald-400"
                  )}
                  style={{ width: `${pred.probability * 100}%` }}
                />
              </div>

              <div className="flex items-center justify-between w-full">
                {uncertaintyStyle ? (
                  <span className={cn(
                    "px-2 py-0.5 rounded text-[10px] font-bold border uppercase tracking-wide",
                    uncertaintyStyle.bg, uncertaintyStyle.text, uncertaintyStyle.border
                  )}>
                    {uncertaintyStyle.label}
                  </span>
                ) : (
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold border uppercase tracking-wide bg-gray-50 text-gray-500 border-gray-200">
                    No uncertainty data
                  </span>
                )}
                <ChevronRight className={cn(
                  "w-4 h-4 transition-transform", 
                  isSelected ? "text-blue-500 translate-x-1" : "text-gray-300 group-hover:text-gray-500"
                )} />
              </div>
            </button>
          );
        })}

        {/* Collapsed below-threshold findings */}
        {belowThreshold.length > 0 && (
          <div className="border-t border-gray-100 pt-2 mt-2">
            <p className="text-[10px] text-gray-400 uppercase tracking-wider font-semibold mb-1 px-1">
              Below threshold ({belowThreshold.length})
            </p>
            {belowThreshold.map((pred) => (
              <button
                key={pred.id}
                onClick={() => setSelectedLabel(pred.label)}
                className={cn(
                  "w-full text-left px-2 py-1.5 rounded text-xs flex items-center justify-between",
                  selectedLabel === pred.label ? "bg-blue-50 text-blue-800" : "text-gray-400 hover:text-gray-600 hover:bg-gray-50"
                )}
              >
                <span className="truncate">{pred.label}</span>
                <span className="font-mono tabular-nums">{(pred.probability * 100).toFixed(1)}%</span>
              </button>
            ))}
          </div>
        )}

        {sortedPredictions.length === 0 && (
          <div className="text-center py-8 text-sm text-gray-500">
            No predictions available for this case.
          </div>
        )}
      </div>
    </div>
  );
}
