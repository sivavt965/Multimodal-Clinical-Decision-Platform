'use client';

import React, { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useCaseStore } from '@/store/caseStore';
import { WorkspaceHeader } from '@/components/case/WorkspaceHeader';
import { CaseTabs } from '@/components/case/CaseTabs';
import { ConsultationSidebar } from '@/components/case/ConsultationSidebar';
import { WorkspaceSkeleton } from '@/components/shared/SkeletonLoaders';

export default function CaseWorkspacePage({ params }: { params: { id: string } }) {
  const { id } = params;
  const router = useRouter();

  const currentCase = useCaseStore((state) => state.currentCase);
  const selectCase  = useCaseStore((state) => state.selectCase);
  const isFetchingCase = useCaseStore((state) => state.isFetchingCase);
  const isSidebarOpen  = useCaseStore((state) => state.isSidebarOpen);
  const closeSidebar   = useCaseStore((state) => state.closeSidebar);

  // Always close sidebar when entering a workspace page
  useEffect(() => { closeSidebar(); }, []);

  // Escape key closes sidebar
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeSidebar(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [closeSidebar]);

  useEffect(() => {
    if (!currentCase || currentCase.case.id !== id) {
      selectCase(id);
    }
  }, [id, selectCase, currentCase]);

  if (isFetchingCase || (!currentCase && isFetchingCase !== false)) {
    return (
      <div className="-mx-4 sm:-mx-6 lg:-mx-8 -my-8 bg-gray-50 h-[calc(100vh-64px)] overflow-hidden flex flex-col">
        <div className="bg-white border-b border-gray-200 h-20 shrink-0" />
        <WorkspaceSkeleton />
      </div>
    );
  }

  if (!currentCase && !isFetchingCase) {
    return (
      <div className="flex flex-col items-center justify-center h-[calc(100vh-64px)] -mx-4 sm:-mx-6 lg:-mx-8 -my-8 bg-gray-50 p-6">
        <h2 className="text-3xl font-bold text-gray-900 mb-3">Case Not Found</h2>
        <p className="text-gray-500 mb-8 text-lg">The case ID &quot;{id}&quot; does not exist or failed to load.</p>
        <button
          onClick={() => router.push('/dashboard')}
          className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2.5 px-6 rounded-lg transition-colors shadow-sm"
        >
          Return to Dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="relative flex -mx-4 sm:-mx-6 lg:-mx-8 -my-8 h-[calc(100vh-64px)] bg-gray-50 overflow-hidden">

      {/* Backdrop — clicking outside the sidebar closes it */}
      {isSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/25 z-40 backdrop-blur-[1px] cursor-pointer"
          onClick={closeSidebar}
        />
      )}

      {/* Main workspace — shifts left when sidebar is open */}
      <div className={`flex-1 flex flex-col transition-all duration-300 ease-in-out ${isSidebarOpen ? 'mr-80' : 'mr-0'}`}>
        <WorkspaceHeader caseDetail={currentCase!} />
        <div className="flex-1 overflow-hidden">
          <CaseTabs />
        </div>
      </div>

      {/* Consultation Sidebar */}
      <ConsultationSidebar />

    </div>
  );
}
