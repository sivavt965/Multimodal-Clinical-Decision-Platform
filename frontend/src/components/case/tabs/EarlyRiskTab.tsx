'use client';

import React from 'react';
import { useCaseStore } from '@/store/caseStore';
import { RISK_BADGE_COLORS, RiskBadge, PhaseARisk, CLINICAL_NORMAL_RANGES } from '@/lib/types';
import { Activity, Beaker, BrainCircuit, AlertTriangle, Info, TrendingUp, ShieldCheck, AlertCircle } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

// MIMIC-IV itemid → human-readable name + unit
const MIMIC_LAB_MAP: Record<string, { name: string; unit: string }> = {
  '50801': { name: 'Alveolar-arterial Gradient', unit: '' },
  '50802': { name: 'Base Excess', unit: 'mEq/L' },
  '50803': { name: 'Calculated Total CO2', unit: 'mEq/L' },
  '50804': { name: 'Calculated Bicarbonate', unit: 'mEq/L' },
  '50813': { name: 'Lactate', unit: 'mmol/L' },
  '50818': { name: 'pCO2', unit: 'mmHg' },
  '50820': { name: 'pH', unit: '' },
  '50821': { name: 'pO2', unit: 'mmHg' },
  '50822': { name: 'Potassium (ABG)', unit: 'mEq/L' },
  '50824': { name: 'Sodium (ABG)', unit: 'mEq/L' },
  '50825': { name: 'Temperature', unit: '°F' },
  '50882': { name: 'Bicarbonate', unit: 'mEq/L' },
  '50893': { name: 'Calcium Total', unit: 'mg/dL' },
  '50902': { name: 'Chloride', unit: 'mEq/L' },
  '50910': { name: 'Creatine Kinase (CK)', unit: 'IU/L' },
  '50912': { name: 'Creatinine', unit: 'mg/dL' },
  '50931': { name: 'Glucose', unit: 'mg/dL' },
  '50956': { name: 'Lipase', unit: 'IU/L' },
  '50960': { name: 'Magnesium', unit: 'mg/dL' },
  '50970': { name: 'Phosphate', unit: 'mg/dL' },
  '50971': { name: 'Potassium', unit: 'mEq/L' },
  '50983': { name: 'Sodium', unit: 'mEq/L' },
  '51006': { name: 'Urea Nitrogen (BUN)', unit: 'mg/dL' },
  '51221': { name: 'Hematocrit', unit: '%' },
  '51222': { name: 'Hemoglobin', unit: 'g/dL' },
  '51248': { name: 'MCV', unit: 'fL' },
  '51249': { name: 'MCH', unit: 'pg' },
  '51250': { name: 'MCHC', unit: 'g/dL' },
  '51256': { name: 'Neutrophils', unit: '%' },
  '51265': { name: 'Platelet Count', unit: 'K/uL' },
  '51274': { name: 'PT', unit: 'sec' },
  '51275': { name: 'PTT', unit: 'sec' },
  '51277': { name: 'RDW', unit: '%' },
  '51279': { name: 'Red Blood Cells', unit: 'M/uL' },
  '51301': { name: 'White Blood Cells', unit: 'K/uL' },
  '51144': { name: 'Bands', unit: '%' },
  '51146': { name: 'Basophils', unit: '%' },
  '51200': { name: 'Eosinophils', unit: '%' },
  '51244': { name: 'Lymphocytes', unit: '%' },
  '51254': { name: 'Monocytes', unit: '%' },
  '50885': { name: 'Bilirubin Total', unit: 'mg/dL' },
  '50878': { name: 'AST', unit: 'IU/L' },
  '50861': { name: 'ALT', unit: 'IU/L' },
  '50863': { name: 'Alkaline Phosphatase', unit: 'IU/L' },
  '51214': { name: 'Fibrinogen', unit: 'mg/dL' },
  '51237': { name: 'INR', unit: '' },
  '50889': { name: 'C-Reactive Protein', unit: 'mg/L' },
  '50907': { name: 'Cortisol', unit: 'ug/dL' },
};

function mapPhaseRiskToBadge(risk: PhaseARisk | null): RiskBadge {
  if (risk === 'High') return 'Elevated Risk';
  if (risk === 'Moderate') return 'Monitor';
  return 'Unlikely';
}

// ── Clinical stat card ────────────────────────────────────────────────
function StatCard({
  label, value, unit, normalRange,
}: { label: string; value: string | number; unit?: string; normalRange?: string }) {
  return (
    <div className="group bg-white border border-gray-100 rounded-lg p-3 hover:border-gray-200 hover:shadow-sm transition-all duration-150">
      <div className="text-[11px] text-gray-500 font-medium mb-1 truncate" title={label}>{label}</div>
      <div className="flex items-baseline gap-1">
        <span className="text-lg font-bold text-gray-900 tabular-nums">{value}</span>
        {unit && <span className="text-[11px] text-gray-500 font-medium">{unit}</span>}
      </div>
      {normalRange && (
        <div className="text-[10px] text-gray-400 mt-1 font-medium">
          Normal: {normalRange}
        </div>
      )}
    </div>
  );
}

// ── Section container ─────────────────────────────────────────────────
function Section({
  title, icon: Icon, iconColor, meta, children,
}: {
  title: string;
  icon: React.ElementType;
  iconColor: string;
  meta?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="bg-white rounded-xl border border-gray-100 shadow-card overflow-hidden animate-fadeInUp">
      <div className="bg-gradient-to-b from-gray-50 to-white border-b border-gray-100 px-5 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className={cn("w-7 h-7 rounded-lg flex items-center justify-center", iconColor)}>
            <Icon className="w-4 h-4" />
          </div>
          <h3 className="font-semibold text-gray-900 text-sm">{title}</h3>
        </div>
        {meta && <span className="text-xs text-gray-500">{meta}</span>}
      </div>
      <div className="p-5">{children}</div>
    </section>
  );
}

export function EarlyRiskTab() {
  const currentCase = useCaseStore((state) => state.currentCase);

  if (!currentCase) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        No case selected.
      </div>
    );
  }

  const { case: caseData } = currentCase;
  const { ecg_data, lab_data } = caseData;

  const riskBadgeType = mapPhaseRiskToBadge(caseData.phase_a_risk_level);
  const badgeStyle = RISK_BADGE_COLORS[riskBadgeType];

  // Risk accent for the right panel
  const riskAccent = riskBadgeType === 'Elevated Risk'
    ? { bar: 'bg-red-500',    bgSoft: 'bg-red-50',    ring: 'ring-red-100',    text: 'text-red-700',    border: 'border-red-200',    Icon: AlertCircle }
    : riskBadgeType === 'Monitor'
      ? { bar: 'bg-amber-400', bgSoft: 'bg-amber-50',  ring: 'ring-amber-100',  text: 'text-amber-700',  border: 'border-amber-200',  Icon: TrendingUp }
      : { bar: 'bg-emerald-500', bgSoft: 'bg-emerald-50', ring: 'ring-emerald-100', text: 'text-emerald-700', border: 'border-emerald-200', Icon: ShieldCheck };
  const RiskIconComp = riskAccent.Icon;

  return (
    <div className="h-full w-full max-w-7xl mx-auto">

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ── Left: Clinical Stats ────────────────────────────── */}
        <div className="lg:col-span-2 space-y-6">

          {/* ECG */}
          {ecg_data && (
            <Section
              title="Electrocardiogram (ECG)"
              icon={Activity}
              iconColor="bg-blue-50 text-blue-600"
              meta={ecg_data.acquired_at ? new Date(ecg_data.acquired_at).toLocaleString() : undefined}
            >
              <div className="mb-4 pb-4 border-b border-gray-100">
                <div className="text-[11px] text-gray-500 font-semibold uppercase tracking-wider mb-2">Rhythm Interpretation</div>
                <div className="inline-flex items-center gap-2 text-sm font-semibold text-blue-800 bg-blue-50 border border-blue-200 px-3 py-1.5 rounded-lg">
                  <Activity className="w-3.5 h-3.5" />
                  {ecg_data.rhythm_interpretation || 'Unknown'}
                </div>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
                {ecg_data.heart_rate       != null && <StatCard label="Heart Rate"   value={ecg_data.heart_rate} unit="bpm" normalRange="60–100" />}
                {ecg_data.pr_interval_ms   != null && <StatCard label="PR Interval"  value={ecg_data.pr_interval_ms} unit="ms" normalRange={CLINICAL_NORMAL_RANGES.pr_interval} />}
                {ecg_data.qrs_duration_ms  != null && <StatCard label="QRS Duration" value={ecg_data.qrs_duration_ms} unit="ms" normalRange={CLINICAL_NORMAL_RANGES.qrs} />}
                {ecg_data.qtc_ms           != null && <StatCard label="QTc"          value={ecg_data.qtc_ms} unit="ms" normalRange={CLINICAL_NORMAL_RANGES.qtc} />}
                {ecg_data.st_deviation_mm  != null && <StatCard label="ST Deviation" value={ecg_data.st_deviation_mm > 0 ? `+${ecg_data.st_deviation_mm}` : ecg_data.st_deviation_mm} unit="mm" normalRange="< 1.0" />}
              </div>
            </Section>
          )}

          {/* Labs */}
          {(lab_data || caseData.labs_raw) && (
            <Section
              title="Laboratory Results"
              icon={Beaker}
              iconColor="bg-purple-50 text-purple-600"
              meta={lab_data?.collected_at ? new Date(lab_data.collected_at).toLocaleString() : undefined}
            >
              <div className="space-y-5">
                {lab_data && (
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400 mb-2">Phase A Summary Panel</p>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                      {lab_data.troponin_ng_ml    != null && <StatCard label="Troponin"    value={lab_data.troponin_ng_ml}    unit="ng/mL"   normalRange={CLINICAL_NORMAL_RANGES.troponin} />}
                      {lab_data.bnp_pg_ml         != null && <StatCard label="BNP"         value={lab_data.bnp_pg_ml}         unit="pg/mL"   normalRange={CLINICAL_NORMAL_RANGES.bnp} />}
                      {lab_data.wbc_count         != null && <StatCard label="WBC Count"   value={lab_data.wbc_count}         unit="x10³/µL" normalRange={CLINICAL_NORMAL_RANGES.wbc} />}
                      {lab_data.creatinine_mg_dl  != null && <StatCard label="Creatinine"  value={lab_data.creatinine_mg_dl}  unit="mg/dL"   normalRange={CLINICAL_NORMAL_RANGES.creatinine} />}
                      {lab_data.sodium_meq_l      != null && <StatCard label="Sodium"      value={lab_data.sodium_meq_l}      unit="mEq/L"   normalRange={CLINICAL_NORMAL_RANGES.sodium} />}
                      {lab_data.potassium_meq_l   != null && <StatCard label="Potassium"   value={lab_data.potassium_meq_l}   unit="mEq/L"   normalRange={CLINICAL_NORMAL_RANGES.potassium} />}
                      {lab_data.lactate_mmol_l    != null && <StatCard label="Lactate"     value={lab_data.lactate_mmol_l}    unit="mmol/L"  normalRange={CLINICAL_NORMAL_RANGES.lactate} />}
                    </div>
                  </div>
                )}

                {caseData.labs_raw && Object.keys(caseData.labs_raw).length > 0 && (() => {
                  const entries = Object.entries(caseData.labs_raw).sort(([a], [b]) => {
                    const nameA = MIMIC_LAB_MAP[a]?.name ?? `itemid ${a}`;
                    const nameB = MIMIC_LAB_MAP[b]?.name ?? `itemid ${b}`;
                    return nameA.localeCompare(nameB);
                  });
                  return (
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Full MIMIC-IV Lab Panel</p>
                        <span className="text-[10px] text-gray-400 font-medium">{entries.length} fields</span>
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5 gap-2">
                        {entries.map(([itemid, value]) => {
                          const meta = MIMIC_LAB_MAP[itemid];
                          const label = meta?.name ?? `Lab ${itemid}`;
                          const unit  = meta?.unit ?? '';
                          const displayValue = typeof value === 'number'
                            ? (Number.isInteger(value) ? value : parseFloat(value.toFixed(3)))
                            : value;
                          return (
                            <StatCard
                              key={itemid}
                              label={label}
                              value={displayValue as string | number}
                              unit={unit || undefined}
                            />
                          );
                        })}
                      </div>
                    </div>
                  );
                })()}
              </div>
            </Section>
          )}

          {!ecg_data && !lab_data && !caseData.labs_raw && (
            <div className="bg-white rounded-xl border border-gray-200 border-dashed p-10 text-center">
              <AlertTriangle className="w-8 h-8 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-500 font-medium">No ECG or Lab data was provided for this case.</p>
            </div>
          )}
        </div>

        {/* ── Right: AI Analysis ────────────────────────────────── */}
        <div className="lg:col-span-1">
          <section className="bg-white rounded-xl border border-gray-100 shadow-card overflow-hidden h-full flex flex-col animate-fadeInUp stagger-1">
            <div className="bg-gradient-to-b from-gray-50 to-white border-b border-gray-100 px-5 py-3 flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-emerald-50 text-emerald-600 flex items-center justify-center">
                <BrainCircuit className="w-4 h-4" />
              </div>
              <h3 className="font-semibold text-gray-900 text-sm">Multimodal AI Analysis</h3>
            </div>

            <div className="p-5 flex-1 flex flex-col">

              {/* Risk indicator */}
              <div className="mb-6">
                <h4 className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-3">Overall Risk Stratification</h4>

                <div className={cn(
                  "rounded-xl border p-4 ring-1",
                  riskAccent.bgSoft, riskAccent.border, riskAccent.ring
                )}>
                  <div className="flex items-center gap-3 mb-3">
                    <div className={cn("w-10 h-10 rounded-lg bg-white flex items-center justify-center shadow-sm border", riskAccent.border)}>
                      <RiskIconComp className={cn("w-5 h-5", riskAccent.text)} />
                    </div>
                    <div>
                      <p className={cn("text-xs font-semibold uppercase tracking-wide", riskAccent.text)}>Risk Level</p>
                      <p className="text-xl font-bold text-gray-900 leading-tight">
                        {caseData.phase_a_risk_level || 'Unknown'}
                      </p>
                    </div>
                  </div>

                  {caseData.phase_a_risk_score !== null && (
                    <>
                      <div className="flex items-center justify-between text-xs font-medium mb-1.5">
                        <span className="text-gray-500">Calculated Score</span>
                        <span className="font-bold text-gray-900 tabular-nums">
                          {(caseData.phase_a_risk_score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full h-2 bg-white/70 rounded-full overflow-hidden ring-1 ring-white">
                        <div
                          className={cn("h-full rounded-full transition-all duration-700 ease-out", riskAccent.bar)}
                          style={{ width: `${caseData.phase_a_risk_score * 100}%` }}
                        />
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Recommendation */}
              <div className="mt-auto">
                <h4 className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">Clinical Recommendation</h4>
                <div className="relative bg-gradient-to-br from-slate-50 to-slate-100/50 border border-slate-200 rounded-xl p-4 pl-5 overflow-hidden">
                  <div className="absolute top-0 left-0 w-1 h-full bg-blue-500" />
                  <div className="flex gap-3">
                    <Info className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
                    <p className="text-sm text-slate-700 font-medium leading-relaxed">
                      {caseData.phase_a_recommendation || 'No recommendation provided.'}
                    </p>
                  </div>
                </div>
              </div>

            </div>
          </section>
        </div>

      </div>
    </div>
  );
}
