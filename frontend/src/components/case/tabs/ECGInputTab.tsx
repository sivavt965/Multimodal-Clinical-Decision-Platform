'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useCaseStore } from '@/store/caseStore';
import { useToastStore } from '@/store/toastStore';
import { updateECGData } from '@/lib/api';
import { CLINICAL_NORMAL_RANGES } from '@/lib/types';
import { Activity, Heart, Save, AlertTriangle, CheckCircle, Zap, Clock } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

/** Debounce hook for live input fields */
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debouncedValue;
}

interface ECGField {
  key: string;
  label: string;
  unit: string;
  normalRange: string;
  icon: React.ReactNode;
  min?: number;
  max?: number;
  step?: number;
}

const ECG_FIELDS: ECGField[] = [
  {
    key: 'heart_rate',
    label: 'Heart Rate',
    unit: 'bpm',
    normalRange: '60–100',
    icon: <Heart className="w-4 h-4" />,
    min: 20, max: 300, step: 1,
  },
  {
    key: 'pr_interval_ms',
    label: 'PR Interval',
    unit: 'ms',
    normalRange: CLINICAL_NORMAL_RANGES.pr_interval,
    icon: <Zap className="w-4 h-4" />,
    min: 50, max: 400, step: 1,
  },
  {
    key: 'qrs_duration_ms',
    label: 'QRS Duration',
    unit: 'ms',
    normalRange: CLINICAL_NORMAL_RANGES.qrs,
    icon: <Activity className="w-4 h-4" />,
    min: 40, max: 300, step: 1,
  },
  {
    key: 'qtc_ms',
    label: 'QTc Interval',
    unit: 'ms',
    normalRange: CLINICAL_NORMAL_RANGES.qtc,
    icon: <Clock className="w-4 h-4" />,
    min: 200, max: 700, step: 1,
  },
  {
    key: 'st_deviation_mm',
    label: 'ST Deviation',
    unit: 'mm',
    normalRange: '±0.5',
    icon: <Activity className="w-4 h-4" />,
    min: -10, max: 10, step: 0.1,
  },
];

const RHYTHM_OPTIONS = [
  'Normal Sinus Rhythm',
  'Atrial Fibrillation',
  'Atrial Flutter',
  'Sinus Tachycardia',
  'Sinus Bradycardia',
  'Supraventricular Tachycardia',
  'Ventricular Tachycardia',
  'First Degree AV Block',
  'Second Degree AV Block',
  'Third Degree AV Block',
  'Left Bundle Branch Block',
  'Right Bundle Branch Block',
  'Premature Ventricular Contractions',
  'Premature Atrial Contractions',
  'Unknown',
];

function isAbnormal(key: string, value: number): boolean {
  switch (key) {
    case 'heart_rate': return value < 60 || value > 100;
    case 'pr_interval_ms': return value < 120 || value > 200;
    case 'qrs_duration_ms': return value >= 120;
    case 'qtc_ms': return value >= 440;
    case 'st_deviation_mm': return Math.abs(value) > 0.5;
    default: return false;
  }
}

export function ECGInputTab() {
  const currentCase = useCaseStore((state) => state.currentCase);
  const selectCase = useCaseStore((state) => state.selectCase);
  const addToast = useToastStore((state) => state.addToast);

  // Initialize from existing ECG data
  const existingECG = currentCase?.case?.ecg_data;

  const [values, setValues] = useState<Record<string, number>>({
    heart_rate: existingECG?.heart_rate ?? 0,
    pr_interval_ms: existingECG?.pr_interval_ms ?? 0,
    qrs_duration_ms: existingECG?.qrs_duration_ms ?? 0,
    qtc_ms: existingECG?.qtc_ms ?? 0,
    st_deviation_mm: existingECG?.st_deviation_mm ?? 0,
  });
  const [rhythm, setRhythm] = useState(existingECG?.rhythm_interpretation ?? 'Normal Sinus Rhythm');
  const [isSaving, setIsSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState<string | null>(null);

  // Sync when case changes
  useEffect(() => {
    if (existingECG) {
      setValues({
        heart_rate: existingECG.heart_rate ?? 0,
        pr_interval_ms: existingECG.pr_interval_ms ?? 0,
        qrs_duration_ms: existingECG.qrs_duration_ms ?? 0,
        qtc_ms: existingECG.qtc_ms ?? 0,
        st_deviation_mm: existingECG.st_deviation_mm ?? 0,
      });
      setRhythm(existingECG.rhythm_interpretation ?? 'Normal Sinus Rhythm');
    }
  }, [existingECG]);

  // Debounce for auto-assessment
  const debouncedValues = useDebounce(values, 500);

  const handleChange = useCallback((key: string, raw: string) => {
    const num = parseFloat(raw);
    setValues(prev => ({ ...prev, [key]: isNaN(num) ? 0 : num }));
  }, []);

  const handleSave = async () => {
    if (!currentCase) return;
    setIsSaving(true);
    try {
      await updateECGData(currentCase.case.id, {
        ...values,
        rhythm_interpretation: rhythm,
      });
      setLastSaved(new Date().toLocaleTimeString());
      addToast({ type: 'success', title: 'ECG Data Saved', message: 'ECG parameters updated successfully.' });
      // Refresh case data
      await selectCase(currentCase.case.id);
    } catch (err: any) {
      addToast({
        type: 'error',
        title: 'ECG Save Failed',
        message: err.message || 'Could not save ECG data.',
      });
    } finally {
      setIsSaving(false);
    }
  };

  // Count abnormal values
  const abnormalCount = ECG_FIELDS.filter(f => {
    const v = debouncedValues[f.key];
    return v > 0 && isAbnormal(f.key, v);
  }).length;
  const isAbnormalRhythm = rhythm !== 'Normal Sinus Rhythm' && rhythm !== 'Unknown';
  const totalAbnormalities = abnormalCount + (isAbnormalRhythm ? 1 : 0);

  if (!currentCase) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        No case selected.
      </div>
    );
  }

  return (
    <div className="h-full w-full max-w-[1000px] mx-auto flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-pink-600 flex items-center justify-center shadow-lg shadow-rose-500/20">
            <Activity className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900">ECG Parameters</h2>
            <p className="text-sm text-gray-500">
              12-Lead ECG structured input • {currentCase.patient.first_name} {currentCase.patient.last_name}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {lastSaved && (
            <span className="text-xs text-gray-400 flex items-center gap-1">
              <CheckCircle className="w-3 h-3 text-emerald-500" />
              Saved {lastSaved}
            </span>
          )}
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="px-4 py-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white text-sm font-semibold rounded-lg flex items-center gap-2 shadow-md shadow-blue-500/20 transition-all disabled:opacity-50"
          >
            <Save className="w-4 h-4" />
            {isSaving ? 'Saving...' : 'Save ECG'}
          </button>
        </div>
      </div>

      {/* Abnormality Summary */}
      {totalAbnormalities > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-amber-900">
              {totalAbnormalities} abnormal finding{totalAbnormalities > 1 ? 's' : ''} detected
            </p>
            <p className="text-xs text-amber-700 mt-1">
              Values outside normal clinical ranges are highlighted in red below.
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ECG Parameters */}
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6">
          <h3 className="text-sm font-bold text-gray-700 uppercase tracking-wider mb-5 flex items-center gap-2">
            <Heart className="w-4 h-4 text-rose-500" />
            Interval Measurements
          </h3>
          <div className="space-y-4">
            {ECG_FIELDS.map(field => {
              const val = values[field.key];
              const abnormal = val > 0 && isAbnormal(field.key, val);
              return (
                <div key={field.key}>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-sm font-medium text-gray-700 flex items-center gap-2">
                      <span className={cn(
                        "w-6 h-6 rounded-md flex items-center justify-center",
                        abnormal
                          ? "bg-red-100 text-red-600"
                          : "bg-gray-100 text-gray-500"
                      )}>
                        {field.icon}
                      </span>
                      {field.label}
                    </label>
                    <span className={cn(
                      "text-xs px-2 py-0.5 rounded-full font-medium",
                      abnormal
                        ? "bg-red-100 text-red-700 border border-red-200"
                        : "bg-gray-100 text-gray-500"
                    )}>
                      Normal: {field.normalRange} {field.unit}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min={field.min}
                      max={field.max}
                      step={field.step}
                      value={val || ''}
                      onChange={e => handleChange(field.key, e.target.value)}
                      placeholder="0"
                      className={cn(
                        "flex-1 border rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none transition-colors",
                        abnormal
                          ? "border-red-300 bg-red-50 text-red-900"
                          : "border-gray-300 bg-white text-gray-900"
                      )}
                    />
                    <span className="text-xs text-gray-400 w-8 text-right">{field.unit}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Rhythm + Info */}
        <div className="flex flex-col gap-6">
          {/* Rhythm Selection */}
          <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6">
            <h3 className="text-sm font-bold text-gray-700 uppercase tracking-wider mb-4 flex items-center gap-2">
              <Zap className="w-4 h-4 text-indigo-500" />
              Rhythm Interpretation
            </h3>
            <select
              value={rhythm}
              onChange={e => setRhythm(e.target.value)}
              className={cn(
                "w-full border rounded-lg px-3 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 outline-none bg-white transition-colors",
                isAbnormalRhythm
                  ? "border-amber-300 text-amber-900"
                  : "border-gray-300 text-gray-900"
              )}
            >
              {RHYTHM_OPTIONS.map(r => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
            {isAbnormalRhythm && (
              <p className="text-xs text-amber-700 mt-2 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" />
                Abnormal rhythm detected
              </p>
            )}
          </div>

          {/* ECG Quick Assessment */}
          <div className="bg-gradient-to-br from-slate-50 to-gray-100 border border-gray-200 rounded-xl shadow-sm p-6 flex-1">
            <h3 className="text-sm font-bold text-gray-700 uppercase tracking-wider mb-4">
              Quick Assessment
            </h3>
            <div className="space-y-3">
              {ECG_FIELDS.map(field => {
                const val = debouncedValues[field.key];
                const abnormal = val > 0 && isAbnormal(field.key, val);
                const filled = val > 0;
                return (
                  <div key={field.key} className="flex items-center justify-between text-sm">
                    <span className="text-gray-600">{field.label}</span>
                    {!filled ? (
                      <span className="text-gray-300 text-xs italic">Not entered</span>
                    ) : abnormal ? (
                      <span className="text-red-600 font-semibold flex items-center gap-1">
                        <AlertTriangle className="w-3 h-3" />
                        {val} {field.unit}
                      </span>
                    ) : (
                      <span className="text-emerald-600 font-medium flex items-center gap-1">
                        <CheckCircle className="w-3 h-3" />
                        {val} {field.unit}
                      </span>
                    )}
                  </div>
                );
              })}
              <div className="border-t border-gray-200 pt-3 mt-3 flex items-center justify-between text-sm">
                <span className="text-gray-600">Rhythm</span>
                <span className={cn(
                  "font-medium",
                  isAbnormalRhythm ? "text-amber-600" : "text-emerald-600"
                )}>
                  {rhythm}
                </span>
              </div>
            </div>
          </div>

          {/* Model Info Notice */}
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 text-xs text-blue-700">
            <p className="font-semibold text-blue-900 mb-1">ℹ️ Current Model: CXR-Only Baseline</p>
            <p>
              The loaded model (<code className="bg-blue-100 px-1 rounded">baseline_best.pt</code>)
              is a DenseNet121 CXR classifier. ECG parameters are stored for display
              and future multimodal (Symile) inference. Raw 12-lead ECG signal input
              will be available when the multimodal model is activated.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
