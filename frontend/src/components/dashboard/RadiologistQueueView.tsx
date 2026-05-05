'use client';

import React, { useEffect, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { useCaseStore } from '@/store/caseStore';
import { DashboardSkeleton } from '@/components/shared/SkeletonLoaders';
import { useUserRole } from '@/lib/userRole';
import {
  AlertCircle, ShieldCheck, Activity, Clock, ArrowRight, Image as ImageIcon, Inbox, MessageSquare,
} from 'lucide-react';
import { CaseSummary } from '@/lib/types';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

const RISK_ORDER: Record<string, number> = { High: 0, Moderate: 1, Low: 2 };

const RISK_CONFIG: Record<string, {
  border: string; chip: string; icon: React.ReactNode; dot: string; label: string;
}> = {
  High:     { border: 'border-l-red-500',     chip: 'bg-red-50 text-red-700 border-red-200',          icon: <AlertCircle className="w-3.5 h-3.5" />,  dot: 'bg-red-500 animate-pulse',  label: 'High Risk'     },
  Moderate: { border: 'border-l-amber-400',   chip: 'bg-amber-50 text-amber-800 border-amber-200',    icon: <Activity className="w-3.5 h-3.5" />,    dot: 'bg-amber-400',              label: 'Moderate'      },
  Low:      { border: 'border-l-emerald-500', chip: 'bg-emerald-50 text-emerald-700 border-emerald-200', icon: <ShieldCheck className="w-3.5 h-3.5" />, dot: 'bg-emerald-500',            label: 'Low Risk'      },
};

function minutesAgo(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  const mins = Math.max(0, Math.round(ms / 60000));
  if (mins < 60) return `${mins} min`;
  const hrs = Math.floor(mins / 60);
  const rem = mins % 60;
  if (hrs < 24) return `${hrs}h ${rem}m`;
  return `${Math.floor(hrs / 24)}d ${hrs % 24}h`;
}

function QueueRow({ summary, index }: { summary: CaseSummary; index: number }) {
  const router = useRouter();
  const setActiveTab = useCaseStore((s) => s.setActiveTab);

  const risk = summary.phase_a_risk_level ?? 'Low';
  const cfg = RISK_CONFIG[risk] ?? RISK_CONFIG.Low;

  const handleStart = () => {
    setActiveTab('cxr-analysis');
    router.push(`/dashboard/case/${summary.case_id}`);
  };

  return (
    <div
      onClick={handleStart}
      className={cn(
        'group relative bg-white rounded-xl border border-gray-100 border-l-4',
        cfg.border,
        'shadow-[0_1px_3px_rgba(0,0,0,0.06)] hover:shadow-[0_6px_20px_rgba(0,0,0,0.10)]',
        'transition-all duration-200 ease-out hover:-translate-y-0.5 cursor-pointer',
        'opacity-0 animate-fadeInUp',
      )}
      style={{ animationDelay: `${index * 60}ms`, animationFillMode: 'both' }}
    >
      <div className="flex items-center gap-4 px-5 py-4">
        {/* Position + risk dot */}
        <div className="shrink-0 w-9 flex flex-col items-center">
          <span className={cn('inline-block w-2.5 h-2.5 rounded-full', cfg.dot)} />
          <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mt-1">#{index + 1}</span>
        </div>

        {/* Patient */}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <p className="font-semibold text-gray-900 text-[15px] leading-tight truncate">{summary.patient_name}</p>
            {summary.reanalysis_requested && (
              <span className="shrink-0 inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide bg-blue-100 text-blue-700 border border-blue-200 animate-pulseSoft">
                <MessageSquare className="w-3 h-3" />
                Reanalysis Requested
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 text-xs text-gray-400 mt-0.5">
            <span className="font-mono tracking-wide tabular-nums">{summary.mrn}</span>
            <span className="w-px h-3 bg-gray-200" />
            <Clock className="w-3 h-3" />
            <span className="tabular-nums">Pending {minutesAgo(summary.admitted_at)}</span>
          </div>
        </div>

        {/* Risk chip */}
        <div className="shrink-0 hidden sm:flex">
          <span className={cn('inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border', cfg.chip)}>
            {cfg.icon}
            {cfg.label}
          </span>
        </div>

        {/* Existing top finding (if any) — hint of prior inference */}
        <div className="shrink-0 hidden md:block w-40 text-right">
          {summary.top_finding_label ? (
            <p className="text-xs text-gray-500 truncate">
              <span className="text-gray-400">Last AI: </span>
              <span className="font-medium text-gray-700">{summary.top_finding_label}</span>
            </p>
          ) : (
            <p className="text-xs text-gray-300 italic">No prior inference</p>
          )}
        </div>

        {/* Start Analysis */}
        <button
          onClick={(e) => { e.stopPropagation(); handleStart(); }}
          className="shrink-0 inline-flex items-center gap-1.5 px-3.5 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-lg shadow-sm transition-all duration-150 group/btn"
        >
          <ImageIcon className="w-3.5 h-3.5" />
          Start Analysis
          <ArrowRight className="w-3.5 h-3.5 group-hover/btn:translate-x-0.5 transition-transform" />
        </button>
      </div>
    </div>
  );
}

export function RadiologistQueueView() {
  const cases = useCaseStore((s) => s.cases);
  const fetchCases = useCaseStore((s) => s.fetchCases);
  const isFetchingCases = useCaseStore((s) => s.isFetchingCases);
  const { user } = useUserRole();

  useEffect(() => { fetchCases(); }, [fetchCases]);

  const queue = useMemo(() => {
    // Triage order: high risk first, then oldest pending first.
    return [...cases].sort((a, b) => {
      const ra = RISK_ORDER[a.phase_a_risk_level ?? ''] ?? 3;
      const rb = RISK_ORDER[b.phase_a_risk_level ?? ''] ?? 3;
      if (ra !== rb) return ra - rb;
      return new Date(a.admitted_at).getTime() - new Date(b.admitted_at).getTime();
    });
  }, [cases]);

  const counts = useMemo(() => ({
    high: cases.filter((c) => c.phase_a_risk_level === 'High').length,
    moderate: cases.filter((c) => c.phase_a_risk_level === 'Moderate').length,
    low: cases.filter((c) => c.phase_a_risk_level === 'Low').length,
  }), [cases]);

  return (
    <div className="space-y-6">

      {/* Header */}
      <div className="flex justify-between items-end border-b border-gray-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">My Queue</h1>
          <p className="text-sm text-gray-500 mt-1">
            Cases awaiting CXR analysis · ordered by clinical priority
          </p>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-500">Signed in as</p>
          <p className="font-semibold text-gray-900">{user.full_name}</p>
        </div>
      </div>

      {/* Quick triage summary */}
      <div className="grid grid-cols-3 gap-4">
        <SummaryPill count={counts.high}     label="High risk"     tint="red"     icon={<AlertCircle className="w-5 h-5" />} />
        <SummaryPill count={counts.moderate} label="Moderate"      tint="amber"   icon={<Activity className="w-5 h-5" />} />
        <SummaryPill count={counts.low}      label="Low risk"      tint="emerald" icon={<ShieldCheck className="w-5 h-5" />} />
      </div>

      {/* Queue */}
      {isFetchingCases && cases.length === 0 ? (
        <DashboardSkeleton />
      ) : queue.length === 0 ? (
        <div className="py-20 flex flex-col items-center gap-3 text-center bg-white rounded-xl border border-gray-200 border-dashed">
          <Inbox className="w-10 h-10 text-gray-300" />
          <div>
            <p className="font-semibold text-gray-700">Queue is clear</p>
            <p className="text-sm text-gray-400 mt-0.5">No pending cases. Ward team will route new ones automatically.</p>
          </div>
        </div>
      ) : (
        <div className="flex flex-col gap-2.5">
          {queue.map((s, i) => <QueueRow key={s.case_id} summary={s} index={i} />)}
        </div>
      )}
    </div>
  );
}

function SummaryPill({
  count, label, tint, icon,
}: { count: number; label: string; tint: 'red' | 'amber' | 'emerald'; icon: React.ReactNode }) {
  const styles = {
    red:     { bg: 'bg-red-50',     text: 'text-red-700',     icon: 'text-red-500'     },
    amber:   { bg: 'bg-amber-50',   text: 'text-amber-800',   icon: 'text-amber-500'   },
    emerald: { bg: 'bg-emerald-50', text: 'text-emerald-700', icon: 'text-emerald-500' },
  }[tint];

  return (
    <div className="bg-white rounded-xl border border-gray-100 shadow-sm px-5 py-4 flex items-center gap-4">
      <div className={cn('w-10 h-10 rounded-xl flex items-center justify-center shrink-0', styles.bg)}>
        <span className={styles.icon}>{icon}</span>
      </div>
      <div className="min-w-0">
        <p className="text-2xl font-bold text-gray-900 leading-tight tabular-nums">{count}</p>
        <p className={cn('text-xs font-medium truncate', styles.text)}>{label}</p>
      </div>
    </div>
  );
}
