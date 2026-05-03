'use client';

import React from 'react';
import { useRouter } from 'next/navigation';
import { ArrowRight, MessageSquare, Trash2, AlertCircle, ShieldCheck, Activity, Clock } from 'lucide-react';
import { CaseSummary, PhaseARisk } from '@/lib/types';
import { useCaseStore } from '@/store/caseStore';

interface CaseRowProps {
  summary: CaseSummary;
  index: number;
}

function getInitials(name: string) {
  return name.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase();
}

function avatarGradient(name: string) {
  const gradients = [
    'from-blue-500 to-blue-600',
    'from-violet-500 to-purple-600',
    'from-emerald-500 to-teal-600',
    'from-orange-500 to-amber-600',
    'from-rose-500 to-pink-600',
    'from-cyan-500 to-sky-600',
    'from-indigo-500 to-blue-700',
    'from-fuchsia-500 to-purple-700',
  ];
  let hash = 0;
  for (let i = 0; i < name.length; i++) hash = name.charCodeAt(i) + ((hash << 5) - hash);
  return gradients[Math.abs(hash) % gradients.length];
}

const RISK_CONFIG: Record<string, {
  border: string; badge: string; icon: React.ReactNode; dot: string;
}> = {
  High: {
    border: 'border-l-red-500',
    badge: 'bg-red-50 text-red-700 border-red-200 ring-red-100',
    icon: <AlertCircle className="w-3.5 h-3.5" />,
    dot: 'bg-red-500 animate-pulse',
  },
  Moderate: {
    border: 'border-l-amber-400',
    badge: 'bg-amber-50 text-amber-700 border-amber-200 ring-amber-100',
    icon: <Activity className="w-3.5 h-3.5" />,
    dot: 'bg-amber-400',
  },
  Low: {
    border: 'border-l-emerald-500',
    badge: 'bg-emerald-50 text-emerald-700 border-emerald-200 ring-emerald-100',
    icon: <ShieldCheck className="w-3.5 h-3.5" />,
    dot: 'bg-emerald-500',
  },
};

export function CaseRow({ summary, index }: CaseRowProps) {
  const router = useRouter();
  const removeCase = useCaseStore((state) => state.removeCase);

  const risk = summary.phase_a_risk_level ?? 'Low';
  const cfg = RISK_CONFIG[risk] ?? RISK_CONFIG.Low;
  const prob = summary.top_finding_probability ?? 0;

  const admissionDate = new Date(summary.admitted_at);
  const dateStr = admissionDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  const timeStr = admissionDate.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

  const handleView = () => router.push(`/dashboard/case/${summary.case_id}`);

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(`Remove ${summary.patient_name}'s case?`)) removeCase(summary.case_id);
  };

  return (
    <div
      className={[
        'group relative bg-white rounded-xl border border-gray-100 border-l-4',
        cfg.border,
        'shadow-[0_1px_3px_rgba(0,0,0,0.08)] hover:shadow-[0_6px_20px_rgba(0,0,0,0.12)]',
        'transition-all duration-200 ease-out hover:-translate-y-0.5',
        'cursor-pointer overflow-hidden',
        'opacity-0 animate-fadeInUp',
      ].join(' ')}
      style={{ animationDelay: `${index * 60}ms`, animationFillMode: 'both' }}
      onClick={handleView}
    >
      <div className="flex items-center gap-4 px-5 py-4">

        {/* Avatar */}
        <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${avatarGradient(summary.patient_name)} flex items-center justify-center text-white font-bold text-sm shrink-0 shadow-sm`}>
          {getInitials(summary.patient_name)}
        </div>

        {/* Patient Info */}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="font-semibold text-gray-900 text-[15px] leading-tight truncate">
              {summary.patient_name}
            </span>
            {summary.urgency_flag && (
              <span className="shrink-0 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase tracking-wide bg-red-100 text-red-600">
                Urgent
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 text-xs text-gray-400">
            <span className="font-mono tracking-wide tabular-nums">{summary.mrn}</span>
            <span className="w-px h-3 bg-gray-200" />
            <Clock className="w-3 h-3" />
            <span className="tabular-nums">{dateStr} · {timeStr}</span>
            {summary.consultation_open && (
              <>
                <span className="w-px h-3 bg-gray-200" />
                <span className="flex items-center gap-1 text-amber-600 font-medium">
                  <MessageSquare className="w-3 h-3" />
                  Consult open
                </span>
              </>
            )}
          </div>
        </div>

        {/* Phase A Risk */}
        <div className="shrink-0 hidden sm:flex items-center">
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ring-1 ${cfg.badge}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
            {cfg.icon}
            {risk} Risk
          </span>
        </div>

        {/* Top Finding */}
        <div className="shrink-0 hidden md:flex flex-col gap-1 w-44">
          {summary.top_finding_label ? (
            <>
              <div className="flex justify-between text-xs text-gray-500 mb-0.5">
                <span className="font-medium text-gray-700 truncate">{summary.top_finding_label}</span>
                <span className="font-semibold text-gray-900 ml-2">{(prob * 100).toFixed(0)}%</span>
              </div>
              <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-blue-500 to-blue-400 transition-all duration-500"
                  style={{ width: `${(prob * 100).toFixed(0)}%` }}
                />
              </div>
            </>
          ) : (
            <span className="text-xs text-gray-400 italic">No inference yet</span>
          )}
        </div>

        {/* Actions */}
        <div className="shrink-0 flex items-center gap-2 ml-2">
          <button
            onClick={handleDelete}
            className="p-2 text-gray-300 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors opacity-0 group-hover:opacity-100"
            title="Remove case"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleView}
            className="flex items-center gap-1.5 px-3.5 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-lg shadow-sm transition-all duration-150 group/btn"
          >
            View
            <ArrowRight className="w-3.5 h-3.5 group-hover/btn:translate-x-0.5 transition-transform" />
          </button>
        </div>
      </div>
    </div>
  );
}
