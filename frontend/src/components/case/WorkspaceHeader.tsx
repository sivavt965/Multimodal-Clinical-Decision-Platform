'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import {
  ArrowLeft, User, Stethoscope, MessageSquare, CheckCircle2,
  Trash2, Lock, RefreshCw, Upload, ImageIcon, Activity, Beaker,
  CheckCircle, AlertCircle, MoreHorizontal,
} from 'lucide-react';
import { CaseDetail } from '@/lib/types';
import { useCaseStore } from '@/store/caseStore';
import { UploadModal } from '@/components/case/UploadModal';
import { useUserRole } from '@/lib/userRole';

interface WorkspaceHeaderProps {
  caseDetail: CaseDetail;
}

// ── Avatar with gradient ──────────────────────────────────────────────
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

export function WorkspaceHeader({ caseDetail }: WorkspaceHeaderProps) {
  const { patient, consultation, predictions } = caseDetail;
  const patientName = `${patient.first_name} ${patient.last_name}`;
  const toggleSidebar       = useCaseStore((state) => state.toggleSidebar);
  const isSidebarOpen       = useCaseStore((state) => state.isSidebarOpen);
  const removeCase          = useCaseStore((state) => state.removeCase);
  const completeCurrentCase = useCaseStore((state) => state.completeCurrentCase);
  const rerunInference      = useCaseStore((state) => state.rerunInference);

  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isDeleting,    setIsDeleting]    = useState(false);
  const [isCompleting,  setIsCompleting]  = useState(false);
  const [isRerunning,   setIsRerunning]   = useState(false);
  const [uploadModal,   setUploadModal]   = useState<'cxr' | 'ecg' | 'labs' | null>(null);

  // Role gating per architecture spec:
  //   ward_doctor  : owns the clinical decision — completes cases, deletes
  //   radiologist  : can re-run inference (delegates Complete Case to ward)
  //   clinical_admin / system_admin : no clinical actions on case detail
  const { role } = useUserRole();
  const canComplete = role === 'ward_doctor';
  const canDelete   = role === 'ward_doctor';
  const canRerun    = role === 'ward_doctor' || role === 'radiologist';

  const isDischarged   = !!caseDetail.case.discharged_at;
  const hasPredictions = predictions?.some(p => p.probability > 0);

  // ── Status badge ────────────────────────────────────────────────────
  const isOpen = consultation?.is_open;
  const statusLabel = isDischarged
    ? 'Discharged'
    : isOpen
      ? 'Consultation Open'
      : hasPredictions
        ? 'Analysis Ready'
        : 'Awaiting Inference';
  const statusColor = isDischarged
    ? 'bg-slate-50 text-slate-600 border-slate-200'
    : isOpen
      ? 'bg-amber-50 text-amber-700 border-amber-200'
      : hasPredictions
        ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
        : 'bg-blue-50 text-blue-700 border-blue-200';
  const StatusIcon = isDischarged ? Lock : isOpen ? MessageSquare : Stethoscope;

  // ── Unread ──────────────────────────────────────────────────────────
  const unreadCount = consultation?.messages?.filter(m => !m.read).length ?? 0;

  // ── Modality completeness ───────────────────────────────────────────
  const hasCXR  = !!caseDetail.case.cxr_dicom_url;
  const hasECG  = !!(caseDetail.case.ecg_data?.heart_rate);
  const hasLabs = !!(caseDetail.case.lab_data && (
    caseDetail.case.lab_data.troponin_ng_ml != null ||
    caseDetail.case.lab_data.wbc_count != null
  ));

  const modalities = [
    { key: 'cxr'  as const, label: 'CXR',  icon: ImageIcon,  has: hasCXR  },
    { key: 'ecg'  as const, label: 'ECG',  icon: Activity,   has: hasECG  },
    { key: 'labs' as const, label: 'Labs', icon: Beaker,     has: hasLabs },
  ];

  // ── Handlers ────────────────────────────────────────────────────────
  const handleComplete = async () => {
    setIsCompleting(true);
    await completeCurrentCase();
    setIsCompleting(false);
  };

  const handleRerun = async () => {
    setIsRerunning(true);
    await rerunInference();
    setIsRerunning(false);
  };

  const handleDelete = async () => {
    setIsDeleting(true);
    await removeCase(caseDetail.case.id);
    setIsDeleting(false);
    setShowDeleteConfirm(false);
  };

  return (
    <>
      <div className="bg-white border-b border-gray-200 px-6 shrink-0 animate-fadeInDown">

        {/* ── Row 1: Identity + Primary Actions ────────────────────── */}
        <div className="py-4 flex flex-col lg:flex-row lg:items-center justify-between gap-4">

          {/* Left: back + avatar + patient */}
          <div className="flex items-center gap-4 min-w-0">
            <Link
              href="/dashboard"
              className="p-2 -ml-2 text-gray-400 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title="Back to Dashboard"
            >
              <ArrowLeft className="w-5 h-5" />
            </Link>

            <div className="flex items-center gap-3 pl-4 border-l border-gray-200 min-w-0">
              <div className={`h-11 w-11 rounded-xl bg-gradient-to-br ${avatarGradient(patientName)} flex items-center justify-center text-white font-bold text-sm shadow-sm shrink-0`}>
                {patient.first_name[0]}{patient.last_name[0]}
              </div>
              <div className="min-w-0">
                <h2 className="text-xl font-bold text-gray-900 leading-tight truncate">{patientName}</h2>
                <div className="flex items-center text-xs text-gray-500 gap-2 mt-0.5">
                  <User className="w-3 h-3" />
                  <span className="font-mono">{patient.mrn}</span>
                  <span className="text-gray-300">•</span>
                  <span>{patient.sex}, {patient.age_at_admission}y</span>
                </div>
              </div>
            </div>

            {/* Status pill */}
            <div className={`hidden md:inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${statusColor} ml-2`}>
              <StatusIcon className="w-3.5 h-3.5" />
              {statusLabel}
            </div>
          </div>

          {/* Right: action buttons */}
          <div className="flex items-center gap-2 flex-wrap ml-12 lg:ml-0">

            {!isDischarged && canRerun && (
              <button
                onClick={handleRerun}
                disabled={isRerunning}
                className="group flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold
                           bg-white border border-gray-200 text-gray-700
                           hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-700
                           shadow-sm transition-all duration-150 disabled:opacity-50"
                title="Re-run Symile inference"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isRerunning ? 'animate-spin' : 'group-hover:rotate-90 transition-transform duration-300'}`} />
                {isRerunning ? 'Running…' : 'Rerun Symile'}
              </button>
            )}

            <button
              onClick={toggleSidebar}
              className={`relative flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold transition-all duration-150 shadow-sm ${
                isSidebarOpen
                  ? 'bg-blue-600 text-white border border-blue-700 hover:bg-blue-700'
                  : 'bg-white text-gray-700 border border-gray-200 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700'
              }`}
            >
              <MessageSquare className="w-3.5 h-3.5" />
              Consultation
              {unreadCount > 0 && !isSidebarOpen && (
                <span className="absolute -top-1.5 -right-1.5 bg-red-500 text-white text-[10px] font-bold min-w-[18px] h-[18px] rounded-full flex items-center justify-center ring-2 ring-white animate-pulseSoft">
                  {unreadCount}
                </span>
              )}
            </button>

            {!isDischarged && canComplete && (
              <button
                onClick={handleComplete}
                disabled={isCompleting}
                className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold
                           bg-emerald-600 hover:bg-emerald-700 text-white
                           shadow-sm transition-all duration-150 disabled:opacity-50"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                {isCompleting ? 'Completing…' : 'Complete Case'}
              </button>
            )}

            {canDelete && (
              <button
                onClick={() => setShowDeleteConfirm(true)}
                className="p-2 rounded-lg text-gray-400 hover:text-red-600 hover:bg-red-50
                           border border-transparent hover:border-red-200
                           transition-all duration-150"
                title="Remove case"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        {/* ── Row 2: Modality indicators ───────────────────────────── */}
        <div className="pb-3 flex items-center gap-3">
          <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider shrink-0">Modalities</span>
          <div className="flex items-center gap-2 flex-wrap">
            {modalities.map(({ key, label, icon: Icon, has }) => (
              <div key={key}>
                {has ? (
                  <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2.5 py-1 rounded-full">
                    <CheckCircle className="w-3 h-3" />
                    <Icon className="w-3 h-3" />
                    {label}
                  </span>
                ) : (
                  <button
                    onClick={() => setUploadModal(key)}
                    className="group inline-flex items-center gap-1.5 text-[11px] font-semibold text-amber-700
                               bg-amber-50 border border-amber-200 px-2.5 py-1 rounded-full
                               hover:bg-amber-100 hover:border-amber-300
                               transition-all duration-150"
                    title={`Upload ${label} data`}
                  >
                    <AlertCircle className="w-3 h-3" />
                    <Icon className="w-3 h-3" />
                    {label}
                    <Upload className="w-3 h-3 ml-0.5 group-hover:-translate-y-0.5 transition-transform" />
                  </button>
                )}
              </div>
            ))}

            {/* Mobile: status pill moves here */}
            <span className={`md:hidden inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold border ${statusColor}`}>
              <StatusIcon className="w-3 h-3" />
              {statusLabel}
            </span>
          </div>
        </div>
      </div>

      {/* ── Upload Modal ─────────────────────────────────────────── */}
      {uploadModal && (
        <UploadModal
          type={uploadModal}
          caseId={caseDetail.case.id}
          onClose={() => setUploadModal(null)}
        />
      )}

      {/* ── Delete Confirmation ──────────────────────────────────── */}
      {showDeleteConfirm && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-fadeIn"
          onClick={() => setShowDeleteConfirm(false)}
        >
          <div
            className="bg-white rounded-2xl shadow-xl p-6 max-w-sm w-full mx-4 animate-scaleIn"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-start gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-red-50 flex items-center justify-center shrink-0">
                <AlertCircle className="w-5 h-5 text-red-600" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-gray-900 leading-tight">Remove case?</h3>
                <p className="text-sm text-gray-600 mt-1">
                  This will permanently delete <strong>{patientName}</strong>&apos;s case and all associated predictions, consultations, and imaging data.
                </p>
              </div>
            </div>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="px-4 py-2 text-sm font-semibold text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={isDeleting}
                className="px-4 py-2 text-sm font-bold text-white bg-red-600 hover:bg-red-700 rounded-lg shadow-sm transition-colors disabled:opacity-50"
              >
                {isDeleting ? 'Deleting…' : 'Delete Permanently'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
