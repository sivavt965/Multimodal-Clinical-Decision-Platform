'use client';

import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { ActivitySquare, Image as ImageIcon, Copy, SplitSquareHorizontal, HeartPulse, Lock } from 'lucide-react';
import { useCaseStore } from '@/store/caseStore';
import { EarlyRiskTab } from '@/components/case/tabs/EarlyRiskTab';
import { CXRAnalysisTab } from '@/components/case/tabs/CXRAnalysisTab';
import { BeforeAfterTab } from '@/components/case/tabs/BeforeAfterTab';
import { SimilarCasesTab } from '@/components/case/tabs/SimilarCasesTab';
import { ECGInputTab } from '@/components/case/tabs/ECGInputTab';
import { useUserRole, ROLE_LABELS } from '@/lib/userRole';
import type { UserRole } from '@/lib/types';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

type TabDef = {
  id: string;
  label: string;
  icon: typeof ActivitySquare;
  /** Roles that may see this tab. */
  roles: UserRole[];
  /** Optional per-role label override — same data, different framing. */
  labelByRole?: Partial<Record<UserRole, string>>;
};

// Tab visibility per role.
//   ward_doctor : full multimodal flow, owns the clinical decision
//   radiologist : CXR-focused; sees Early Risk read-only as Patient Summary,
//                 hidden from Similar Cases / Before-After (ward-doctor tools)
//   clinical_admin / system_admin : no case-detail tabs (own different surfaces)
const TABS: TabDef[] = [
  {
    id: 'early-risk',
    label: 'Early Risk',
    icon: ActivitySquare,
    roles: ['ward_doctor', 'radiologist'],
    // Radiologist: this is their reference-only view of demographics + ECG + labs.
    labelByRole: { radiologist: 'Patient Summary' },
  },
  { id: 'cxr-analysis',  label: 'CXR Analysis',      icon: ImageIcon,             roles: ['ward_doctor', 'radiologist'] },
  { id: 'ecg-input',     label: 'ECG Input',         icon: HeartPulse,            roles: ['ward_doctor'] },
  { id: 'similar-cases', label: 'Similar Cases',     icon: Copy,                  roles: ['ward_doctor', 'radiologist'] },
  { id: 'before-after',  label: 'Before vs. After',  icon: SplitSquareHorizontal, roles: ['ward_doctor', 'radiologist'] },
];

export function CaseTabs() {
  const activeTab    = useCaseStore((state) => state.activeTab);
  const setActiveTab = useCaseStore((state) => state.setActiveTab);
  const navRef       = useRef<HTMLDivElement | null>(null);
  const tabRefs      = useRef<Record<string, HTMLButtonElement | null>>({});
  const [indicator, setIndicator] = useState<{ left: number; width: number }>({ left: 0, width: 0 });
  const { role, hydrated } = useUserRole();

  // Tabs the current role is allowed to see
  const visibleTabs = useMemo(
    () => TABS.filter((t) => t.roles.includes(role)),
    [role]
  );

  // If the persisted active tab is not visible for this role, snap to the first one.
  useEffect(() => {
    if (!hydrated) return;
    if (visibleTabs.length === 0) return;
    if (!visibleTabs.some((t) => t.id === activeTab)) {
      setActiveTab(visibleTabs[0].id);
    }
  }, [hydrated, role, activeTab, visibleTabs, setActiveTab]);

  // Measure the active tab to position the sliding underline
  useLayoutEffect(() => {
    const el = tabRefs.current[activeTab];
    if (!el || !navRef.current) return;
    const navRect = navRef.current.getBoundingClientRect();
    const elRect  = el.getBoundingClientRect();
    setIndicator({
      left:  elRect.left - navRect.left + navRef.current.scrollLeft,
      width: elRect.width,
    });
  }, [activeTab, visibleTabs.length]);

  // Roles without any case-detail tabs see a friendly empty state instead.
  if (hydrated && visibleTabs.length === 0) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-50">
        <div className="max-w-sm text-center px-6 py-10">
          <div className="mx-auto w-12 h-12 rounded-2xl bg-slate-100 flex items-center justify-center mb-4">
            <Lock className="w-6 h-6 text-slate-400" />
          </div>
          <p className="font-semibold text-slate-700">
            Case detail isn&apos;t part of the {ROLE_LABELS[role]} workflow.
          </p>
          <p className="text-sm text-slate-500 mt-2">
            Switch to the Ward Doctor or Radiologist role from the header to view this case.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gray-50 overflow-hidden">

      {/* Tab Navigation */}
      <div className="bg-white border-b border-gray-200 px-6 shrink-0 overflow-x-auto">
        <nav ref={navRef} className="relative flex gap-1" aria-label="Tabs">
          {visibleTabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            const label = tab.labelByRole?.[role] ?? tab.label;
            return (
              <button
                key={tab.id}
                ref={(el) => { tabRefs.current[tab.id] = el; }}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "group relative inline-flex items-center gap-2 py-4 px-3 font-medium text-sm whitespace-nowrap",
                  "transition-colors duration-200",
                  isActive
                    ? "text-blue-600"
                    : "text-gray-500 hover:text-gray-900"
                )}
              >
                <Icon className={cn(
                  "w-4 h-4 transition-colors duration-200",
                  isActive ? "text-blue-600" : "text-gray-400 group-hover:text-gray-600"
                )} />
                {label}
              </button>
            );
          })}

          {/* Animated active underline */}
          <span
            aria-hidden
            className="absolute bottom-0 h-0.5 bg-blue-600 rounded-full transition-all duration-300 ease-out"
            style={{
              left:  `${indicator.left}px`,
              width: `${indicator.width}px`,
            }}
          />
        </nav>
      </div>

      {/* Tab Content — keyed so each mount runs its fade-in */}
      <div className="flex-1 overflow-y-auto p-6">
        <div
          key={activeTab}
          className="h-full animate-fadeInUp"
        >
          {activeTab === 'early-risk'    && <EarlyRiskTab />}
          {activeTab === 'cxr-analysis'  && <CXRAnalysisTab />}
          {activeTab === 'ecg-input'     && <ECGInputTab />}
          {activeTab === 'similar-cases' && <SimilarCasesTab />}
          {activeTab === 'before-after'  && <BeforeAfterTab />}
        </div>
      </div>
    </div>
  );
}
