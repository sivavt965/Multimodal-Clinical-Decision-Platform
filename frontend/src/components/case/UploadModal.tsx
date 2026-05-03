'use client';

import React, { useState, useRef } from 'react';
import { useCaseStore } from '@/store/caseStore';
import { useToastStore } from '@/store/toastStore';
import { X, Upload, ImageIcon, Activity, Beaker, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface UploadModalProps {
  type: 'cxr' | 'ecg' | 'labs';
  caseId: string;
  onClose: () => void;
}

const MODAL_CONFIG = {
  cxr: {
    title: 'Upload Chest X-Ray',
    icon: ImageIcon,
    color: 'text-blue-600',
    bg: 'bg-blue-50',
    border: 'border-blue-200',
    accept: '.png,.jpg,.jpeg,.dcm',
    description: 'Upload a CXR image. After upload, Phase B inference (Grad-CAM + predictions) will run automatically.',
    acceptLabel: 'PNG, JPEG, DICOM — max 10 MB',
  },
  ecg: {
    title: 'Upload ECG Data',
    icon: Activity,
    color: 'text-emerald-600',
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
    accept: '.json,.csv',
    description: 'Upload structured ECG measurements (JSON or CSV). Phase A risk will be recalculated.',
    acceptLabel: 'JSON or CSV — fields: heart_rate, pr_interval_ms, qrs_duration_ms, qtc_ms, st_deviation_mm, rhythm_interpretation',
  },
  labs: {
    title: 'Upload Lab Results',
    icon: Beaker,
    color: 'text-purple-600',
    bg: 'bg-purple-50',
    border: 'border-purple-200',
    accept: '.json,.csv',
    description: 'Upload lab results (JSON or CSV). Phase A risk will be recalculated with the new values.',
    acceptLabel: 'JSON or CSV — keys can be MIMIC-IV itemids or lab names',
  },
};

type UploadStatus = 'idle' | 'uploading' | 'success' | 'error';

export function UploadModal({ type, caseId, onClose }: UploadModalProps) {
  const selectCase = useCaseStore((state) => state.selectCase);
  const addToast   = useToastStore((state) => state.addToast);

  const cfg = MODAL_CONFIG[type];
  const Icon = cfg.icon;

  const [file, setFile]           = useState<File | null>(null);
  const [status, setStatus]       = useState<UploadStatus>('idle');
  const [errorMsg, setErrorMsg]   = useState('');
  const [dragOver, setDragOver]   = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // ECG manual form fields (alternative to file upload for ECG)
  const [ecgForm, setEcgForm] = useState({
    heart_rate: '',
    pr_interval_ms: '',
    qrs_duration_ms: '',
    qtc_ms: '',
    st_deviation_mm: '',
    rhythm_interpretation: 'Normal Sinus Rhythm',
  });
  const [ecgMode, setEcgMode] = useState<'file' | 'form'>('form');

  const handleFile = (f: File) => {
    if (type === 'cxr' && f.size > 10 * 1024 * 1024) {
      setErrorMsg('File too large — maximum 10 MB');
      return;
    }
    setFile(f);
    setErrorMsg('');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  const handleUpload = async () => {
    setStatus('uploading');
    setErrorMsg('');

    try {
      let res: Response;

      if (type === 'cxr') {
        if (!file) { setErrorMsg('Please select a file.'); setStatus('idle'); return; }
        const form = new FormData();
        form.append('image', file);
        res = await fetch(`${API_BASE}/api/cases/${caseId}/upload/cxr`, { method: 'POST', body: form });

      } else if (type === 'ecg') {
        if (ecgMode === 'form') {
          const body: Record<string, string | number> = {
            rhythm_interpretation: ecgForm.rhythm_interpretation,
          };
          if (ecgForm.heart_rate)       body.heart_rate        = parseFloat(ecgForm.heart_rate);
          if (ecgForm.pr_interval_ms)   body.pr_interval_ms    = parseFloat(ecgForm.pr_interval_ms);
          if (ecgForm.qrs_duration_ms)  body.qrs_duration_ms   = parseFloat(ecgForm.qrs_duration_ms);
          if (ecgForm.qtc_ms)           body.qtc_ms            = parseFloat(ecgForm.qtc_ms);
          if (ecgForm.st_deviation_mm)  body.st_deviation_mm   = parseFloat(ecgForm.st_deviation_mm);

          res = await fetch(`${API_BASE}/api/cases/${caseId}/upload/ecg`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          });
        } else {
          if (!file) { setErrorMsg('Please select a file.'); setStatus('idle'); return; }
          const form = new FormData();
          form.append('file', file);
          res = await fetch(`${API_BASE}/api/cases/${caseId}/upload/ecg-file`, { method: 'POST', body: form });
        }

      } else {
        if (!file) { setErrorMsg('Please select a file.'); setStatus('idle'); return; }
        const form = new FormData();
        form.append('file', file);
        res = await fetch(`${API_BASE}/api/cases/${caseId}/upload/labs`, { method: 'POST', body: form });
      }

      if (!res.ok) {
        const body = await res.text().catch(() => '');
        throw new Error(body || `Server returned ${res.status}`);
      }

      setStatus('success');
      addToast({
        type: 'success',
        title: `${cfg.title} — Uploaded`,
        message: type === 'cxr'
          ? 'CXR saved. Phase B inference is running in the background.'
          : 'Data saved. Phase A risk has been recalculated.',
      });

      // Refresh the case in the store after a short delay
      setTimeout(async () => {
        await selectCase(caseId);
        onClose();
      }, 1200);

    } catch (err: any) {
      setStatus('error');
      setErrorMsg(err.message || 'Upload failed');
      addToast({ type: 'error', title: 'Upload Failed', message: err.message || 'Please try again.' });
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4" onClick={onClose}>
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg flex flex-col overflow-hidden" onClick={(e) => e.stopPropagation()}>

        {/* Header */}
        <div className={cn("px-6 py-4 flex items-center justify-between border-b", cfg.bg, cfg.border)}>
          <div className="flex items-center gap-3">
            <div className={cn("p-2 rounded-lg bg-white border", cfg.border)}>
              <Icon className={cn("w-5 h-5", cfg.color)} />
            </div>
            <div>
              <h2 className="text-base font-bold text-gray-900">{cfg.title}</h2>
              <p className="text-xs text-gray-500">{cfg.description}</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-white/60 rounded-lg transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 flex flex-col gap-5">

          {/* ECG: toggle form vs file */}
          {type === 'ecg' && (
            <div className="flex bg-gray-100 rounded-lg p-0.5 gap-0.5">
              <button
                type="button"
                onClick={() => setEcgMode('form')}
                className={cn("flex-1 py-1.5 rounded-md text-xs font-semibold transition-all",
                  ecgMode === 'form' ? "bg-white text-gray-800 shadow-sm" : "text-gray-500")}
              >
                Manual Entry
              </button>
              <button
                type="button"
                onClick={() => setEcgMode('file')}
                className={cn("flex-1 py-1.5 rounded-md text-xs font-semibold transition-all",
                  ecgMode === 'file' ? "bg-white text-gray-800 shadow-sm" : "text-gray-500")}
              >
                Upload File
              </button>
            </div>
          )}

          {/* ECG manual form */}
          {type === 'ecg' && ecgMode === 'form' && (
            <div className="grid grid-cols-2 gap-3">
              {[
                { key: 'heart_rate',       label: 'Heart Rate',      unit: 'bpm',  placeholder: '72' },
                { key: 'pr_interval_ms',   label: 'PR Interval',     unit: 'ms',   placeholder: '160' },
                { key: 'qrs_duration_ms',  label: 'QRS Duration',    unit: 'ms',   placeholder: '88' },
                { key: 'qtc_ms',           label: 'QTc',             unit: 'ms',   placeholder: '420' },
                { key: 'st_deviation_mm',  label: 'ST Deviation',    unit: 'mm',   placeholder: '0.5' },
              ].map(({ key, label, unit, placeholder }) => (
                <div key={key}>
                  <label className="text-xs font-semibold text-gray-500 block mb-1">{label} <span className="font-normal text-gray-400">({unit})</span></label>
                  <input
                    type="number"
                    step="any"
                    placeholder={placeholder}
                    value={ecgForm[key as keyof typeof ecgForm]}
                    onChange={(e) => setEcgForm(prev => ({ ...prev, [key]: e.target.value }))}
                    className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
                  />
                </div>
              ))}
              <div className="col-span-2">
                <label className="text-xs font-semibold text-gray-500 block mb-1">Rhythm Interpretation</label>
                <select
                  value={ecgForm.rhythm_interpretation}
                  onChange={(e) => setEcgForm(prev => ({ ...prev, rhythm_interpretation: e.target.value }))}
                  className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 bg-white"
                >
                  {['Normal Sinus Rhythm','Atrial Fibrillation','Atrial Flutter','Sinus Bradycardia',
                    'Sinus Tachycardia','ST Elevation','ST Depression','Left Bundle Branch Block',
                    'Right Bundle Branch Block','Ventricular Tachycardia','Other'].map(r => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>
              </div>
            </div>
          )}

          {/* File drop zone — shown for CXR, Labs, and ECG file mode */}
          {(type !== 'ecg' || ecgMode === 'file') && (
            <div
              className={cn(
                "border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all",
                dragOver ? "border-blue-400 bg-blue-50/50 scale-[1.01]" : "border-gray-200 hover:border-gray-300 hover:bg-gray-50"
              )}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
            >
              <input
                ref={fileRef}
                type="file"
                accept={cfg.accept}
                className="hidden"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
              />
              {file ? (
                <div className="flex flex-col items-center gap-2">
                  <CheckCircle className="w-8 h-8 text-emerald-500" />
                  <p className="text-sm font-semibold text-gray-700">{file.name}</p>
                  <p className="text-xs text-gray-400">{(file.size / 1024).toFixed(1)} KB — click to change</p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2">
                  <Upload className="w-8 h-8 text-gray-300" />
                  <p className="text-sm font-semibold text-gray-600">Drop file here or click to browse</p>
                  <p className="text-xs text-gray-400 max-w-[280px]">{cfg.acceptLabel}</p>
                </div>
              )}
            </div>
          )}

          {/* Error */}
          {errorMsg && (
            <div className="flex items-center gap-2 text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              {errorMsg}
            </div>
          )}

          {/* Success */}
          {status === 'success' && (
            <div className="flex items-center gap-2 text-sm text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-2">
              <CheckCircle className="w-4 h-4 shrink-0" />
              Upload successful — refreshing case data…
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 pb-6 flex justify-end gap-3">
          <button onClick={onClose} className="px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-lg transition-colors">
            Cancel
          </button>
          <button
            onClick={handleUpload}
            disabled={status === 'uploading' || status === 'success' || (type !== 'ecg' && !file) || (type === 'ecg' && ecgMode === 'file' && !file)}
            className={cn(
              "flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-bold text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed",
              type === 'cxr'  ? "bg-blue-600 hover:bg-blue-700" :
              type === 'ecg'  ? "bg-emerald-600 hover:bg-emerald-700" :
                                "bg-purple-600 hover:bg-purple-700"
            )}
          >
            {status === 'uploading' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
            {status === 'uploading' ? 'Uploading…' : 'Upload & Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
