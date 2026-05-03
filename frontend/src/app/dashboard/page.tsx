'use client';

import React, { useEffect, useState, useMemo } from 'react';
import { useCaseStore } from '@/store/caseStore';
import { CaseRow } from '@/components/dashboard/CaseRow';
import { DashboardSkeleton } from '@/components/shared/SkeletonLoaders';
import { NewCaseWizard } from '@/components/dashboard/NewCaseWizard';
import { RadiologistQueueView } from '@/components/dashboard/RadiologistQueueView';
import { ClinicalAdminView } from '@/components/dashboard/ClinicalAdminView';
import { useUserRole } from '@/lib/userRole';
import {
  UserPlus, Search, SlidersHorizontal, AlertCircle,
  Users, MessageSquare, Zap, ChevronDown, X, FileX,
} from 'lucide-react';

// ── Stat card ────────────────────────────────────────────────────────────────
function StatCard({
  label, value, icon, accent,
}: { label: string; value: number; icon: React.ReactNode; accent: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-100 shadow-[0_1px_3px_rgba(0,0,0,0.07)] px-5 py-4 flex items-center gap-4">
      <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${accent}`}>
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-2xl font-bold text-gray-900 leading-tight tabular-nums">{value}</p>
        <p className="text-xs text-gray-500 font-medium truncate">{label}</p>
      </div>
    </div>
  );
}

// ── Risk ordering for sort ────────────────────────────────────────────────────
const RISK_ORDER: Record<string, number> = { High: 0, Moderate: 1, Low: 2 };

// ── Page ─────────────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const cases          = useCaseStore((state) => state.cases);
  const fetchCases     = useCaseStore((state) => state.fetchCases);
  const isFetchingCases = useCaseStore((state) => state.isFetchingCases);
  const { role, hydrated } = useUserRole();
  const [isWizardOpen, setIsWizardOpen] = useState(false);
  const [search,       setSearch]       = useState('');
  const [riskFilter,   setRiskFilter]   = useState<'All' | 'High' | 'Moderate' | 'Low'>('All');
  const [sortBy,       setSortBy]       = useState<'date' | 'name' | 'risk'>('date');

  useEffect(() => { fetchCases(); }, [fetchCases]);

  // Derived stats — must run unconditionally (rules of hooks)
  const stats = useMemo(() => ({
    total:    cases.length,
    high:     cases.filter(c => c.phase_a_risk_level === 'High').length,
    consults: cases.filter(c => c.consultation_open).length,
    urgent:   cases.filter(c => c.urgency_flag).length,
  }), [cases]);

  // Filter + sort
  const filtered = useMemo(() => {
    let list = [...cases];

    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(c =>
        c.patient_name.toLowerCase().includes(q) ||
        c.mrn.toLowerCase().includes(q)
      );
    }

    if (riskFilter !== 'All') {
      list = list.filter(c => c.phase_a_risk_level === riskFilter);
    }

    list.sort((a, b) => {
      if (sortBy === 'date') {
        return new Date(b.admitted_at).getTime() - new Date(a.admitted_at).getTime();
      }
      if (sortBy === 'name') {
        return a.patient_name.localeCompare(b.patient_name);
      }
      // risk
      const ra = RISK_ORDER[a.phase_a_risk_level ?? ''] ?? 3;
      const rb = RISK_ORDER[b.phase_a_risk_level ?? ''] ?? 3;
      return ra - rb;
    });

    return list;
  }, [cases, search, riskFilter, sortBy]);

  const hasData = !isFetchingCases || cases.length > 0;

  // Role-based view switch — placed AFTER all hooks to keep hook order stable
  // across renders (rules of hooks). The early returns above were causing
  // "Rendered fewer hooks than expected" when switching roles.
  if (hydrated && role === 'radiologist')    return <RadiologistQueueView />;
  if (hydrated && role === 'clinical_admin') return <ClinicalAdminView />;

  return (
    <div className="space-y-6">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="flex justify-between items-end border-b border-gray-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 tracking-tight">Active Patient Cases</h1>
          <p className="text-sm text-gray-500 mt-1">
            Select a patient to enter the multimodal analysis workspace.
          </p>
        </div>
        <button
          onClick={() => setIsWizardOpen(true)}
          className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-lg font-medium text-sm transition-colors flex items-center gap-2 shadow-sm shrink-0"
        >
          <UserPlus className="w-4 h-4" />
          Register New Patient
        </button>
      </div>

      {/* ── Stats Grid ──────────────────────────────────────────────────────── */}
      {hasData && cases.length > 0 && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Total Cases" value={stats.total}
            icon={<Users className="w-5 h-5 text-blue-600" />}
            accent="bg-blue-50"
          />
          <StatCard
            label="High Risk" value={stats.high}
            icon={<AlertCircle className="w-5 h-5 text-red-500" />}
            accent="bg-red-50"
          />
          <StatCard
            label="Open Consultations" value={stats.consults}
            icon={<MessageSquare className="w-5 h-5 text-amber-600" />}
            accent="bg-amber-50"
          />
          <StatCard
            label="Urgent Flags" value={stats.urgent}
            icon={<Zap className="w-5 h-5 text-violet-600" />}
            accent="bg-violet-50"
          />
        </div>
      )}

      {/* ── Search + Filter Bar ─────────────────────────────────────────────── */}
      {hasData && cases.length > 0 && (
        <div className="flex flex-col sm:flex-row gap-3">

          {/* Search */}
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
            <input
              type="text"
              placeholder="Search by name or MRN…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full pl-9 pr-9 py-2.5 text-sm border border-gray-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 transition-colors"
            />
            {search && (
              <button
                onClick={() => setSearch('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {/* Risk Filter */}
          <div className="relative shrink-0">
            <select
              value={riskFilter}
              onChange={e => setRiskFilter(e.target.value as typeof riskFilter)}
              className="appearance-none pl-3 pr-8 py-2.5 text-sm border border-gray-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 transition-colors cursor-pointer"
            >
              <option value="All">All Risk Levels</option>
              <option value="High">High Risk</option>
              <option value="Moderate">Moderate Risk</option>
              <option value="Low">Low Risk</option>
            </select>
            <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400 pointer-events-none" />
          </div>

          {/* Sort */}
          <div className="relative shrink-0">
            <SlidersHorizontal className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400 pointer-events-none" />
            <select
              value={sortBy}
              onChange={e => setSortBy(e.target.value as typeof sortBy)}
              className="appearance-none pl-9 pr-8 py-2.5 text-sm border border-gray-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 transition-colors cursor-pointer"
            >
              <option value="date">Sort: Admitted Date</option>
              <option value="name">Sort: Patient Name</option>
              <option value="risk">Sort: Risk Level</option>
            </select>
            <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400 pointer-events-none" />
          </div>
        </div>
      )}

      {/* ── Case List ───────────────────────────────────────────────────────── */}
      {isFetchingCases && cases.length === 0 ? (
        <DashboardSkeleton />
      ) : (
        <div className="flex flex-col gap-2.5">

          {filtered.map((summary, i) => (
            <CaseRow key={summary.case_id} summary={summary} index={i} />
          ))}

          {/* Empty — no cases at all */}
          {cases.length === 0 && (
            <div className="py-20 flex flex-col items-center gap-4 text-center bg-white rounded-xl border border-gray-200 border-dashed">
              <div className="w-16 h-16 rounded-2xl bg-gray-50 flex items-center justify-center">
                <FileX className="w-8 h-8 text-gray-300" />
              </div>
              <div>
                <p className="font-semibold text-gray-700">No active cases</p>
                <p className="text-sm text-gray-400 mt-1">Register a patient to get started.</p>
              </div>
              <button
                onClick={() => setIsWizardOpen(true)}
                className="mt-1 px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
              >
                Register First Patient
              </button>
            </div>
          )}

          {/* Empty — filters matched nothing */}
          {cases.length > 0 && filtered.length === 0 && (
            <div className="py-12 flex flex-col items-center gap-3 text-center bg-white rounded-xl border border-gray-100">
              <Search className="w-8 h-8 text-gray-300" />
              <div>
                <p className="font-medium text-gray-600">No cases match your filters</p>
                <p className="text-sm text-gray-400 mt-0.5">Try adjusting your search or risk level filter.</p>
              </div>
              <button
                onClick={() => { setSearch(''); setRiskFilter('All'); }}
                className="text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors"
              >
                Clear filters
              </button>
            </div>
          )}

          {/* Result count footer */}
          {filtered.length > 0 && (
            <p className="text-xs text-gray-400 text-center pt-1">
              Showing {filtered.length} of {cases.length} {cases.length === 1 ? 'case' : 'cases'}
            </p>
          )}
        </div>
      )}

      <NewCaseWizard isOpen={isWizardOpen} onClose={() => setIsWizardOpen(false)} />
    </div>
  );
}
