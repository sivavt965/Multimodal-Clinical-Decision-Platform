'use client';

import React from 'react';
import { useRouter } from 'next/navigation';
import { Calendar, User, FileText, ArrowRight, MessageSquare, Activity, Trash2 } from 'lucide-react';
import { CaseSummary, RISK_BADGE_COLORS, RiskBadge, PhaseARisk } from '@/lib/types';
import { useCaseStore } from '@/store/caseStore';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface CaseCardProps {
  summary: CaseSummary;
}

/** Utility to combine tailwind classes */
function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

/** Helper to map Phase A Risk to the standard RiskBadge mapping for colors */
function mapPhaseRiskToBadge(risk: PhaseARisk | null): RiskBadge {
  if (risk === 'High') return 'Elevated Risk';
  if (risk === 'Moderate') return 'Monitor';
  return 'Unlikely';
}

export function CaseCard({ summary }: CaseCardProps) {
  const router = useRouter();
  const removeCase = useCaseStore((state) => state.removeCase);

  // Format Admission Date
  const admissionDate = new Date(summary.admitted_at).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });

  const riskBadgeType = mapPhaseRiskToBadge(summary.phase_a_risk_level);
  const badgeStyle = RISK_BADGE_COLORS[riskBadgeType];

  const handleViewWorkspace = () => {
    router.push(`/dashboard/case/${summary.case_id}`);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(`Remove ${summary.patient_name}'s case?`)) {
      removeCase(summary.case_id);
    }
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md transition-shadow duration-200 overflow-hidden flex flex-col h-full group">
      <div className="p-5 flex-1">
        {/* Header: Patient Info */}
        <div className="flex justify-between items-start mb-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-700 font-bold shrink-0">
              {summary.patient_name.split(' ').map(n => n[0]).join('')}
            </div>
            <div className="min-w-0">
              <h3 className="text-lg font-semibold text-gray-900 leading-none mb-1.5 truncate">
                {summary.patient_name}
              </h3>
              <div className="flex items-center text-sm text-gray-500 gap-1.5">
                <User className="w-3.5 h-3.5 shrink-0" />
                <span className="truncate">MRN: {summary.mrn}</span>
              </div>
            </div>
          </div>
          <button
            onClick={handleDelete}
            className="p-1.5 text-gray-300 hover:text-red-500 hover:bg-red-50 rounded-md transition-colors opacity-0 group-hover:opacity-100"
            title="Remove case"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Body: Case Details */}
        <div className="space-y-2.5">
          <div className="flex items-center text-sm text-gray-600 gap-2">
            <Calendar className="w-4 h-4 text-gray-400 shrink-0" />
            <span className="truncate">{admissionDate}</span>
          </div>

          <div className="flex items-center text-sm text-gray-600 gap-2">
            <FileText className="w-4 h-4 text-gray-400 shrink-0" />
            <span>Phase A:</span>
            <span className={cn(
              "px-2 py-0.5 rounded-full text-xs font-medium border",
              badgeStyle.bg,
              badgeStyle.text,
              badgeStyle.border
            )}>
              {summary.phase_a_risk_level || 'Unknown'}
            </span>
          </div>

          {/* Top AI Finding */}
          {summary.top_finding_label && (
            <div className="flex items-center text-sm text-gray-600 gap-2">
              <Activity className="w-4 h-4 text-gray-400 shrink-0" />
              <span className="truncate">
                {summary.top_finding_label}
                {summary.top_finding_probability != null && (
                  <span className="text-gray-400 ml-1">
                    ({(summary.top_finding_probability * 100).toFixed(0)}%)
                  </span>
                )}
              </span>
            </div>
          )}

          {/* Consultation Status */}
          {summary.consultation_open && (
            <div className="flex items-center text-sm gap-2">
              <MessageSquare className="w-4 h-4 text-amber-500 shrink-0" />
              <span className="text-amber-700 font-medium">Consultation Open</span>
            </div>
          )}
        </div>
      </div>
      
      {/* Footer: Action */}
      <div className="px-5 py-3.5 bg-gray-50 border-t border-gray-100 mt-auto">
        <button
          onClick={handleViewWorkspace}
          className="w-full flex items-center justify-center gap-2 bg-white text-blue-600 border border-blue-200 hover:bg-blue-50 font-medium py-2 px-4 rounded-lg transition-colors duration-150 text-sm"
        >
          View Workspace
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
