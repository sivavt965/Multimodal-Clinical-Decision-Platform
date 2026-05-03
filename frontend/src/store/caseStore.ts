import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import type { CaseDetail, CaseSummary, ViewportState, AnnotationMark, ConsultationMessage } from '@/lib/types';
import { fetchCaseSummaries, fetchCaseDetail, postConsultationMessage, deleteCase, completeCase, reinferCase } from '@/lib/api';
import { useToastStore } from '@/store/toastStore';

type ActiveTab = 'early-risk' | 'cxr-analysis' | string;

type CaseStore = {
  /** The currently selected case (null if none selected) */
  currentCase: CaseDetail | null;
  /** All loaded case summaries */
  cases: CaseSummary[];
  isFetchingCases: boolean;
  isFetchingCase: boolean;
  
  /** Current workspace tab identifier */
  activeTab: ActiveTab;
  /** The CXR label whose Grad-CAM overlay is shown */
  selectedLabel: string | null;
  /** Whether the Grad-CAM overlay is currently visible */
  showOverlay: boolean;
  /** Pan Offset for the image viewer */
  panOffset: { x: number; y: number };
  /** Curtain Slider Position (0-100) */
  sliderPosition: number;
  /** Probability threshold for filtering predictions (0-1) */
  probabilityThreshold: number;
  /** CXR view mode: 'curtain' or 'side-by-side' */
  cxrViewMode: 'curtain' | 'side-by-side';
  /** Whether the Consultation Sidebar is open */
  isSidebarOpen: boolean;

  /** Actions */
  fetchCases: () => Promise<void>;
  selectCase: (caseId: string) => Promise<void>;
  sendMessage: (message: ConsultationMessage) => Promise<void>;
  addCase: (newCase: CaseSummary) => void;
  removeCase: (caseId: string) => Promise<void>;
  completeCurrentCase: () => Promise<void>;
  rerunInference: () => Promise<void>;
  refreshCurrentCase: () => Promise<void>;

  setActiveTab: (tab: ActiveTab) => void;
  setSelectedLabel: (label: string | null) => void;
  toggleOverlay: () => void;
  setBrightness: (val: number) => void;
  setContrast: (val: number) => void;
  setZoom: (val: number) => void;
  setPanOffset: (offset: { x: number; y: number } | ((prev: {x: number, y: number}) => {x: number, y: number})) => void;
  setSliderPosition: (pos: number) => void;
  setProbabilityThreshold: (val: number) => void;
  setCxrViewMode: (mode: 'curtain' | 'side-by-side') => void;
  toggleSidebar: () => void;
  closeSidebar: () => void;
  resetViewport: () => void;
  updateViewport: (updates: Partial<ViewportState>) => void;
  addAnnotation: (annotation: AnnotationMark) => void;
};

/**
 * Central Zustand store for the Clinical Decision Support Platform.
 */
export const useCaseStore = create<CaseStore>()(
  devtools((set, get) => ({
    currentCase: null,
    cases: [],
    isFetchingCases: false,
    isFetchingCase: false,
    
    activeTab: 'early-risk',
    selectedLabel: null,
    showOverlay: true,
    panOffset: { x: 0, y: 0 },
    sliderPosition: 50,
    probabilityThreshold: 0.5,
    cxrViewMode: 'curtain' as const,
    isSidebarOpen: false,

    // ------- Async Actions --------------------------------------------------
    fetchCases: async () => {
      set({ isFetchingCases: true });
      try {
        const summaries = await fetchCaseSummaries();
        set({ cases: summaries, isFetchingCases: false });
      } catch (err: any) {
        set({ isFetchingCases: false });
        useToastStore.getState().addToast({
          type: 'error',
          title: 'Failed to load cases',
          message: err.message,
        });
      }
    },

    selectCase: async (caseId: string) => {
      set({ isFetchingCase: true, isSidebarOpen: false, selectedLabel: null, activeTab: 'early-risk' });
      try {
        const detail = await fetchCaseDetail(caseId);
        set({ currentCase: detail, isFetchingCase: false });
      } catch (err: any) {
        set({ isFetchingCase: false });
        useToastStore.getState().addToast({
          type: 'error',
          title: 'Failed to load case',
          message: err.message,
        });
      }
    },

    sendMessage: async (message: ConsultationMessage) => {
      const { currentCase } = get();
      if (!currentCase) return;
      
      try {
        await postConsultationMessage(currentCase.case.id, message);
        
        // Optimistically update the store
        const consultation = currentCase.consultation;
        if (consultation) {
          const updatedConsultation = {
            ...consultation,
            messages: [...consultation.messages, message],
            updated_at: message.sent_at
          };
          set({ currentCase: { ...currentCase, consultation: updatedConsultation } });
        } else {
           // Create consultation structure if it doesn't exist
           const updatedConsultation = {
             id: `cons-auto-${currentCase.case.id.substring(0,8)}`,
             case_id: currentCase.case.id,
             ward_doctor_id: message.role === 'ward_doctor' ? 'doc-auto' : 'doc-auto',
             radiologist_id: message.role === 'radiologist' ? 'rad-auto' : null,
             is_open: true,
             opened_at: message.sent_at,
             closed_at: null,
             urgency_flag: false,
             messages: [message],
             viewport_state: {
               zoom: 1, contrast: 100, brightness: 100, window_center: 40, window_width: 400, annotations: []
             },
             ward_doctor_last_view: null,
             radiologist_last_view: null,
             created_at: message.sent_at,
             updated_at: message.sent_at
           };
           set({ currentCase: { ...currentCase, consultation: updatedConsultation as any } });
        }
      } catch (err: any) {
        useToastStore.getState().addToast({
          type: 'error',
          title: 'Failed to send message',
          message: err.message,
        });
      }
    },

    removeCase: async (caseId: string) => {
      try {
        await deleteCase(caseId);
        set((state) => ({
          cases: state.cases.filter(c => c.case_id !== caseId),
          currentCase: state.currentCase?.case.id === caseId ? null : state.currentCase,
        }));
        useToastStore.getState().addToast({
          type: 'success',
          title: 'Case Removed',
          message: 'The case and all associated data have been deleted.',
        });
      } catch (err: any) {
        useToastStore.getState().addToast({
          type: 'error',
          title: 'Delete Failed',
          message: err.message,
        });
      }
    },

    completeCurrentCase: async () => {
      const { currentCase } = get();
      if (!currentCase) return;
      try {
        const result = await completeCase(currentCase.case.id);
        // Update the current case in store
        set((state) => {
          const updatedCase = state.currentCase ? {
            ...state.currentCase,
            case: { ...state.currentCase.case, discharged_at: result.discharged_at },
          } : null;
          return { currentCase: updatedCase };
        });
        useToastStore.getState().addToast({
          type: 'success',
          title: 'Case Completed',
          message: 'Patient has been discharged and the case archived.',
        });
      } catch (err: any) {
        useToastStore.getState().addToast({
          type: 'error',
          title: 'Complete Failed',
          message: err.message,
        });
      }
    },

    rerunInference: async () => {
      const { currentCase } = get();
      if (!currentCase) return;
      try {
        useToastStore.getState().addToast({
          type: 'info',
          title: 'Inference Started',
          message: 'Re-running Symile model inference...',
        });
        await reinferCase(currentCase.case.id);
        // Reload case detail WITHOUT resetting the active tab
        const detail = await fetchCaseDetail(currentCase.case.id);
        set({ currentCase: detail });
        useToastStore.getState().addToast({
          type: 'success',
          title: 'Inference Complete',
          message: 'Case data has been updated with new predictions.',
        });
      } catch (err: any) {
        useToastStore.getState().addToast({
          type: 'error',
          title: 'Inference Failed',
          message: err.message,
        });
      }
    },

    refreshCurrentCase: async () => {
      const { currentCase } = get();
      if (!currentCase) return;
      try {
        const detail = await fetchCaseDetail(currentCase.case.id);
        set({ currentCase: detail });
      } catch {
        // silent — caller shows its own toast
      }
    },

    // ------- Sync Actions ---------------------------------------------------
    addCase: (newCase) => set((state) => ({ cases: [newCase, ...state.cases] })),
    setActiveTab: (tab) => set({ activeTab: tab }),
    setSelectedLabel: (label) => set({ selectedLabel: label, showOverlay: true }),
    toggleOverlay: () => set((state) => ({ showOverlay: !state.showOverlay })),
    setBrightness: (val) => get().updateViewport({ brightness: val }),
    setContrast: (val) => get().updateViewport({ contrast: val }),
    setZoom: (val) => get().updateViewport({ zoom: val }),
    setPanOffset: (offset) => set((state) => ({ 
      panOffset: typeof offset === 'function' ? offset(state.panOffset) : offset 
    })),
    setSliderPosition: (pos) => set({ sliderPosition: pos }),
    setProbabilityThreshold: (val: number) => set({ probabilityThreshold: val }),
    setCxrViewMode: (mode: 'curtain' | 'side-by-side') => set({ cxrViewMode: mode }),
    toggleSidebar: () => set((state) => ({ isSidebarOpen: !state.isSidebarOpen })),
  closeSidebar: () => set({ isSidebarOpen: false }),
    resetViewport: () => {
      get().updateViewport({ zoom: 1, brightness: 100, contrast: 100 });
      set({ panOffset: { x: 0, y: 0 }, sliderPosition: 50 });
    },

    updateViewport: (updates) => {
      const { currentCase } = get();
      if (!currentCase) return;
      const consultation = currentCase.consultation;

      // Build or update the viewport state
      const baseViewport: ViewportState = consultation?.viewport_state ?? {
        zoom: 1, contrast: 100, brightness: 100,
        window_center: 40, window_width: 400, annotations: [],
      };
      const newViewport: ViewportState = { ...baseViewport, ...updates };

      const updatedConsultation = consultation
        ? { ...consultation, viewport_state: newViewport }
        : {
            id: `vp-auto-${currentCase.case.id.substring(0, 8)}`,
            case_id: currentCase.case.id,
            ward_doctor_id: 'doc-auto',
            radiologist_id: null,
            is_open: true,
            opened_at: new Date().toISOString(),
            closed_at: null,
            urgency_flag: false,
            messages: [],
            viewport_state: newViewport,
            ward_doctor_last_view: null,
            radiologist_last_view: null,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          };

      set({ currentCase: { ...currentCase, consultation: updatedConsultation as any } });
    },

    addAnnotation: (annotation) => {
      const { currentCase } = get();
      if (!currentCase) return;
      const consultation = currentCase.consultation;
      if (!consultation) return;
      
      const newAnnotations = [...consultation.viewport_state.annotations, annotation];
      const newViewport: ViewportState = {
        ...consultation.viewport_state,
        annotations: newAnnotations,
      };
      
      const updatedConsultation = {
        ...consultation,
        viewport_state: newViewport,
      };
      
      set({ currentCase: { ...currentCase, consultation: updatedConsultation } });
    },
  }))
);
