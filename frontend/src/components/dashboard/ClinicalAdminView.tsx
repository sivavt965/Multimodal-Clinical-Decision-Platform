'use client';

import React, { useEffect, useMemo, useState } from 'react';
import {
  ClipboardList, UserPlus, Upload, ListChecks, Search, Clock,
  FileText, Image as ImageIcon, HeartPulse, CheckCircle2, AlertCircle, Loader2,
} from 'lucide-react';
import { useCaseStore } from '@/store/caseStore';
import { useUserRole } from '@/lib/userRole';
import { NewCaseWizard } from '@/components/dashboard/NewCaseWizard';
import { DashboardSkeleton } from '@/components/shared/SkeletonLoaders';
import { uploadCXR, uploadLabs } from '@/lib/api';
import type { CaseSummary } from '@/lib/types';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

type SubTab = 'register' | 'upload' | 'status';

const SUB_TABS: Array<{ id: SubTab; label: string; icon: typeof UserPlus }> = [
  { id: 'register', label: 'Register Patient', icon: UserPlus },
  { id: 'upload',   label: 'Upload Data',      icon: Upload },
  { id: 'status',   label: 'Case Status',      icon: ListChecks },
];

/**
 * Status derived from a case's current state — no clinical reasoning surfaces.
 * Clinical Admins must not see findings, similarity, or predictions per the
 * RBAC spec; this function only inspects metadata fields.
 */
function deriveStatus(c: CaseSummary): { label: string; tone: 'gray' | 'blue' | 'amber' | 'emerald' } {
  if (!c.cxr_dicom_url)        return { label: 'Pending Registration',    tone: 'gray'    };
  if (!c.top_finding_label)    return { label: 'Awaiting CXR Analysis',   tone: 'blue'    };
  if (c.consultation_open)     return { label: 'Awaiting Ward Review',    tone: 'amber'   };
  return                              { label: 'Completed',                tone: 'emerald' };
}

const TONE_STYLES: Record<'gray' | 'blue' | 'amber' | 'emerald', string> = {
  gray:    'bg-gray-100 text-gray-700 border-gray-200',
  blue:    'bg-blue-50 text-blue-700 border-blue-200',
  amber:   'bg-amber-50 text-amber-800 border-amber-200',
  emerald: 'bg-emerald-50 text-emerald-700 border-emerald-200',
};

// ──────────────────────────────────────────────────────────────────────
// Sub-views
// ──────────────────────────────────────────────────────────────────────

function RegisterTab() {
  const [open, setOpen] = useState(false);
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-8">
      <div className="flex items-start gap-5">
        <div className="w-14 h-14 rounded-2xl bg-blue-50 flex items-center justify-center shrink-0">
          <UserPlus className="w-7 h-7 text-blue-600" />
        </div>
        <div className="flex-1">
          <h2 className="text-lg font-semibold text-gray-900">Register a new patient</h2>
          <p className="text-sm text-gray-500 mt-1 max-w-prose">
            Four-step intake: demographics, lab values, ECG metadata, and the chest X-ray.
            Once submitted, the case is queued for the radiologist on duty.
          </p>
          <button
            onClick={() => setOpen(true)}
            className="mt-5 inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg font-medium text-sm transition-colors shadow-sm"
          >
            <UserPlus className="w-4 h-4" />
            Open registration wizard
          </button>
        </div>
      </div>
      <NewCaseWizard isOpen={open} onClose={() => setOpen(false)} />
    </div>
  );
}

type UploadKind = 'cxr' | 'labs';

const UPLOAD_KINDS: Array<{
  id: UploadKind;
  label: string;
  hint: string;
  icon: typeof ImageIcon;
  accept: string;
}> = [
  { id: 'cxr',  label: 'Chest X-ray',   hint: 'PNG · JPEG (max 10 MB)', icon: ImageIcon, accept: 'image/png,image/jpeg' },
  { id: 'labs', label: 'Lab results',   hint: 'CSV or JSON',             icon: FileText,  accept: '.csv,.json,application/json,text/csv' },
];

type UploadResult =
  | { state: 'idle' }
  | { state: 'uploading' }
  | { state: 'success'; message: string }
  | { state: 'error';   message: string };

function UploadTab() {
  const cases = useCaseStore((s) => s.cases);
  const fetchCases = useCaseStore((s) => s.fetchCases);

  const [caseId, setCaseId] = useState('');
  const [kind, setKind] = useState<UploadKind>('cxr');
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<UploadResult>({ state: 'idle' });

  useEffect(() => { if (cases.length === 0) fetchCases(); }, [cases.length, fetchCases]);

  const selectedCase = useMemo(() => cases.find((c) => c.case_id === caseId), [cases, caseId]);
  const activeKind = UPLOAD_KINDS.find((k) => k.id === kind)!;

  const canSubmit = !!caseId && !!file && result.state !== 'uploading';

  const reset = () => {
    setFile(null);
    setResult({ state: 'idle' });
  };

  const handleSubmit = async () => {
    if (!caseId || !file) return;
    setResult({ state: 'uploading' });
    try {
      if (kind === 'cxr')  await uploadCXR(caseId, file);
      if (kind === 'labs') await uploadLabs(caseId, file);
      setResult({ state: 'success', message: `${activeKind.label} replaced for ${selectedCase?.patient_name ?? 'this case'}.` });
      setFile(null);
    } catch (err: any) {
      setResult({ state: 'error', message: err?.message || 'Upload failed. Try again.' });
    }
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-8 max-w-3xl">
      <div className="flex items-start gap-5 mb-6">
        <div className="w-14 h-14 rounded-2xl bg-violet-50 flex items-center justify-center shrink-0">
          <Upload className="w-7 h-7 text-violet-600" />
        </div>
        <div className="flex-1">
          <h2 className="text-lg font-semibold text-gray-900">Replace patient data</h2>
          <p className="text-sm text-gray-500 mt-1">
            Re-upload a chest X-ray or lab file to an existing case. The new file replaces the previous one and re-runs Phase A scoring as needed.
          </p>
        </div>
      </div>

      {/* 1. Pick case */}
      <div className="space-y-1.5">
        <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500">1. Patient case</label>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
          <select
            value={caseId}
            onChange={(e) => { setCaseId(e.target.value); reset(); }}
            className="w-full pl-9 pr-9 py-2.5 text-sm border border-gray-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 appearance-none"
          >
            <option value="">Select a case…</option>
            {cases.map((c) => (
              <option key={c.case_id} value={c.case_id}>
                {c.patient_name} · {c.mrn} · {c.case_id.slice(0, 8)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* 2. Pick kind */}
      <div className="space-y-1.5 mt-5">
        <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500">2. File type</label>
        <div className="grid grid-cols-2 gap-3">
          {UPLOAD_KINDS.map((k) => {
            const Icon = k.icon;
            const active = kind === k.id;
            return (
              <button
                key={k.id}
                type="button"
                onClick={() => { setKind(k.id); reset(); }}
                className={cn(
                  'flex items-start gap-3 px-4 py-3 rounded-lg border text-left transition-all',
                  active
                    ? 'border-blue-400 bg-blue-50 ring-2 ring-blue-500/15'
                    : 'border-gray-200 bg-white hover:border-gray-300'
                )}
              >
                <Icon className={cn('w-5 h-5 mt-0.5 shrink-0', active ? 'text-blue-600' : 'text-gray-400')} />
                <div>
                  <p className={cn('text-sm font-semibold', active ? 'text-blue-700' : 'text-gray-800')}>{k.label}</p>
                  <p className="text-[11px] text-gray-500">{k.hint}</p>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* 3. Pick file */}
      <div className="space-y-1.5 mt-5">
        <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500">3. File</label>
        <label className={cn(
          'flex flex-col items-center justify-center gap-2 px-6 py-8 border-2 border-dashed rounded-xl cursor-pointer transition-all',
          file ? 'border-emerald-300 bg-emerald-50/40' : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50/30'
        )}>
          <input
            type="file"
            accept={activeKind.accept}
            className="hidden"
            onChange={(e) => { setFile(e.target.files?.[0] ?? null); setResult({ state: 'idle' }); }}
          />
          {file ? (
            <>
              <CheckCircle2 className="w-8 h-8 text-emerald-500" />
              <p className="text-sm font-medium text-emerald-700">{file.name}</p>
              <p className="text-xs text-emerald-600">{(file.size / 1024).toFixed(1)} KB · click to replace</p>
            </>
          ) : (
            <>
              <Upload className="w-7 h-7 text-gray-400" />
              <p className="text-sm font-medium text-gray-600">Click to choose a {activeKind.label.toLowerCase()} file</p>
              <p className="text-xs text-gray-400">{activeKind.hint}</p>
            </>
          )}
        </label>
      </div>

      {/* Submit */}
      <div className="mt-6 flex items-center gap-3">
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className={cn(
            'inline-flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-sm transition-all shadow-sm',
            canSubmit
              ? 'bg-blue-600 hover:bg-blue-700 text-white'
              : 'bg-gray-200 text-gray-400 cursor-not-allowed'
          )}
        >
          {result.state === 'uploading' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
          {result.state === 'uploading' ? 'Uploading…' : 'Upload Replacement'}
        </button>

        {result.state === 'success' && (
          <span className="inline-flex items-center gap-1.5 text-sm text-emerald-700">
            <CheckCircle2 className="w-4 h-4" />
            {result.message}
          </span>
        )}
        {result.state === 'error' && (
          <span className="inline-flex items-center gap-1.5 text-sm text-red-700">
            <AlertCircle className="w-4 h-4" />
            {result.message}
          </span>
        )}
      </div>
    </div>
  );
}

function StatusTab() {
  const cases = useCaseStore((s) => s.cases);
  const fetchCases = useCaseStore((s) => s.fetchCases);
  const isFetching = useCaseStore((s) => s.isFetchingCases);
  const [search, setSearch] = useState('');

  useEffect(() => { fetchCases(); }, [fetchCases]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return q
      ? cases.filter((c) => c.patient_name.toLowerCase().includes(q) || c.mrn.toLowerCase().includes(q))
      : cases;
  }, [cases, search]);

  if (isFetching && cases.length === 0) return <DashboardSkeleton />;

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">

      <div className="px-5 py-4 border-b border-gray-100 flex items-center gap-3">
        <Search className="w-4 h-4 text-gray-400" />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by name or MRN…"
          className="flex-1 text-sm bg-transparent focus:outline-none"
        />
        <span className="text-xs text-gray-400 tabular-nums">{filtered.length} of {cases.length}</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-[11px] font-semibold uppercase tracking-wider text-gray-400 bg-gray-50">
              <th className="px-5 py-2.5">Case ID</th>
              <th className="px-5 py-2.5">Patient</th>
              <th className="px-5 py-2.5">MRN</th>
              <th className="px-5 py-2.5">Admitted</th>
              <th className="px-5 py-2.5">Risk</th>
              <th className="px-5 py-2.5">Status</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((c, i) => {
              const status = deriveStatus(c);
              const dt = new Date(c.admitted_at);
              return (
                <tr
                  key={c.case_id}
                  className={cn('border-t border-gray-100', i % 2 === 1 && 'bg-gray-50/40')}
                >
                  <td className="px-5 py-3 font-mono text-xs text-gray-500 tabular-nums">{c.case_id.slice(0, 8)}</td>
                  <td className="px-5 py-3 font-medium text-gray-900">{c.patient_name}</td>
                  <td className="px-5 py-3 font-mono text-xs text-gray-500 tabular-nums">{c.mrn}</td>
                  <td className="px-5 py-3 text-xs text-gray-500 tabular-nums">
                    <Clock className="w-3 h-3 inline mr-1 text-gray-400" />
                    {dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                  </td>
                  <td className="px-5 py-3 text-xs">
                    <span className="text-gray-700">{c.phase_a_risk_level ?? '—'}</span>
                  </td>
                  <td className="px-5 py-3">
                    <span className={cn('inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border', TONE_STYLES[status.tone])}>
                      {status.label}
                    </span>
                  </td>
                </tr>
              );
            })}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="px-5 py-12 text-center text-sm text-gray-400">
                  No cases match your search.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────────
// Container
// ──────────────────────────────────────────────────────────────────────

export function ClinicalAdminView() {
  const { user } = useUserRole();
  const [tab, setTab] = useState<SubTab>('register');

  return (
    <div className="space-y-6">

      {/* Header */}
      <div className="flex justify-between items-end border-b border-gray-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight flex items-center gap-2">
            <ClipboardList className="w-6 h-6 text-blue-600" />
            Patient Intake
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Register patients, upload supporting data, and monitor case status.
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-500">Signed in as</p>
          <p className="font-semibold text-gray-900">{user.full_name}</p>
        </div>
      </div>

      {/* Sub-tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex gap-1">
          {SUB_TABS.map(({ id, label, icon: Icon }) => {
            const active = tab === id;
            return (
              <button
                key={id}
                onClick={() => setTab(id)}
                className={cn(
                  'relative inline-flex items-center gap-2 py-3 px-4 text-sm font-medium transition-colors',
                  active ? 'text-blue-600' : 'text-gray-500 hover:text-gray-900'
                )}
              >
                <Icon className={cn('w-4 h-4', active ? 'text-blue-600' : 'text-gray-400')} />
                {label}
                {active && <span className="absolute -bottom-px left-3 right-3 h-0.5 bg-blue-600 rounded-full" />}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Body */}
      <div className="animate-fadeInUp">
        {tab === 'register' && <RegisterTab />}
        {tab === 'upload'   && <UploadTab />}
        {tab === 'status'   && <StatusTab />}
      </div>
    </div>
  );
}
