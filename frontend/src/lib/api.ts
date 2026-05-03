// =============================================================================
// api.ts — Centralized API client for the FastAPI backend
// =============================================================================

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/** Custom error class that carries the HTTP status code */
export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/**
 * Generic fetch wrapper with error handling.
 * Throws `ApiError` on non-OK responses or network failures.
 */
/**
 * Read the active dev role + user_id from localStorage so every API call
 * carries identity headers. Pre-auth shim: replaced by Supabase JWT in Phase 5.
 */
function authHeaders(): Record<string, string> {
  if (typeof window === 'undefined') return {};
  const role = window.localStorage.getItem('cdss_dev_role') || 'ward_doctor';
  const ID_BY_ROLE: Record<string, string> = {
    radiologist:    '4f9b9bc8-bfd7-4b3f-924a-0dae0c882f90',
    ward_doctor:    'c29a01e9-f3e6-4a1e-8aff-35a11d49b57c',
    clinical_admin: '5a351665-167b-429f-b6c9-635237995e0f',
    system_admin:   '586531d7-daf0-48f5-80e9-dbbfdfcfcc4b',
  };
  return {
    'X-User-Id':   ID_BY_ROLE[role] ?? '',
    'X-User-Role': role,
  };
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      ...options,
    });
  } catch {
    // Network error — backend is offline
    throw new ApiError(
      'Unable to reach the backend server. Is it running on port 8000?',
      0
    );
  }

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(
      body || `Request failed with status ${res.status}`,
      res.status
    );
  }

  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Typed API methods
// ---------------------------------------------------------------------------
import type { CaseDetail, CaseSummary, ConsultationMessage, PlatformUser, AuditLogEntry } from '@/lib/types';

// ---------------------------------------------------------------------------
// Admin (system_admin only — gating happens client-side until Phase 5)
// ---------------------------------------------------------------------------
export async function listUsers(): Promise<PlatformUser[]> {
  return request<PlatformUser[]>('/api/admin/users');
}

export function createUser(input: { email: string; full_name: string; role: string }): Promise<PlatformUser> {
  return request<PlatformUser>('/api/admin/users', {
    method: 'POST',
    body: JSON.stringify(input),
  });
}

export function updateUser(
  userId: string,
  patch: { role?: string; status?: 'active' | 'inactive' | 'suspended' },
): Promise<PlatformUser> {
  return request<PlatformUser>(`/api/admin/users/${userId}`, {
    method: 'PATCH',
    body: JSON.stringify(patch),
  });
}

export async function listAuditLog(opts: {
  limit?: number;
  offset?: number;
  action?: string;
  userId?: string;
} = {}): Promise<{ total: number; items: AuditLogEntry[] }> {
  const qs = new URLSearchParams();
  if (opts.limit  !== undefined) qs.set('limit',  String(opts.limit));
  if (opts.offset !== undefined) qs.set('offset', String(opts.offset));
  if (opts.action)               qs.set('action', opts.action);
  if (opts.userId)               qs.set('user_id', opts.userId);
  const q = qs.toString();
  return request<{ total: number; items: AuditLogEntry[] }>(`/api/admin/audit-log${q ? `?${q}` : ''}`);
}
export type { CaseDetail };

/** GET /api/cases — all case summaries for the dashboard */
export function fetchCaseSummaries(): Promise<CaseSummary[]> {
  return request<CaseSummary[]>('/api/cases');
}

/** POST /api/cases — register a new case via multipart form data */
export async function createCase(data: FormData): Promise<CaseSummary> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/cases`, {
      method: 'POST',
      body: data,
      // Don't set Content-Type — the browser must add the multipart boundary.
      headers: authHeaders(),
    });
  } catch {
    throw new ApiError('Unable to reach the backend server.', 0);
  }

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(body || `Request failed with status ${res.status}`, res.status);
  }

  return res.json() as Promise<CaseSummary>;
}

/** GET /api/cases/{id} — full multimodal detail for a single case */
export function fetchCaseDetail(caseId: string): Promise<CaseDetail> {
  return request<CaseDetail>(`/api/cases/${caseId}`);
}

/** POST /api/consultation/{id} — append a chat message */
export function postConsultationMessage(
  caseId: string,
  message: ConsultationMessage
): Promise<{ status: string; message_id: string; thread_length: number }> {
  return request(`/api/consultation/${caseId}`, {
    method: 'POST',
    body: JSON.stringify(message),
  });
}

/** GET /api/health — lightweight health check */
export function checkHealth(): Promise<{ status: string; cases_loaded: number }> {
  return request('/api/health');
}

/** POST /api/labs/parse — parse a CSV/JSON lab file via the backend */
export async function parseLabFile(file: File): Promise<{
  status: string;
  lab_count: number;
  labs: Record<string, number>;
  labs_percentile_vector: Record<string, number>;
}> {
  const form = new FormData();
  form.append('file', file);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/labs/parse`, {
      method: 'POST',
      body: form,
    });
  } catch {
    throw new ApiError('Unable to reach the backend server.', 0);
  }

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(body || `Request failed with status ${res.status}`, res.status);
  }

  return res.json();
}

/** POST /api/cases/{id}/ecg — update ECG data for a case */
export function updateECGData(
  caseId: string,
  ecgData: Record<string, number | string>
): Promise<{ status: string; case_id: string; ecg_data: Record<string, unknown> }> {
  return request(`/api/cases/${caseId}/ecg`, {
    method: 'POST',
    body: JSON.stringify(ecgData),
  });
}

/** POST /api/cases/{id}/reinfer — re-run inference for a case */
export function reinferCase(
  caseId: string,
  targetLabel: string = 'Pleural Effusion'
): Promise<{ status: string; case_id: string; message: string }> {
  return request(`/api/cases/${caseId}/reinfer?target_label=${encodeURIComponent(targetLabel)}`, {
    method: 'POST',
  });
}

/** POST /api/cases/{id}/gradcam/regenerate — regenerate Grad-CAM heatmap */
export function regenerateGradCam(
  caseId: string,
  targetLabel: string = 'Pleural Effusion'
): Promise<{ status: string; case_id: string; message: string }> {
  return request(`/api/cases/${caseId}/gradcam/regenerate?target_label=${encodeURIComponent(targetLabel)}`, {
    method: 'POST',
  });
}

/**
 * GET /api/cases/{id} with timeout protection.
 * Aborts the request after timeoutMs (default 15s) and throws an ApiError.
 */
export async function fetchCaseDetailWithTimeout(
  caseId: string,
  timeoutMs: number = 60000
): Promise<CaseDetail> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_BASE}/api/cases/${caseId}`, {
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new ApiError(body || `Request failed with status ${res.status}`, res.status);
    }
    return res.json() as Promise<CaseDetail>;
  } catch (err: any) {
    clearTimeout(timer);
    if (err.name === 'AbortError') {
      throw new ApiError('Request timed out. The server may be busy with inference.', 408);
    }
    if (err instanceof ApiError) throw err;
    throw new ApiError('Unable to reach the backend server.', 0);
  }
}

/** DELETE /api/cases/{id} — remove a case and all associated data */
export async function deleteCase(caseId: string): Promise<{ status: string; deleted: boolean }> {
  return request(`/api/cases/${caseId}`, { method: 'DELETE' });
}

/** POST /api/cases/{id}/flag — radiologist marks a critical finding */
export function flagCaseCritical(
  caseId: string,
  finding: string,
  note?: string,
): Promise<{ status: string; case_id: string; urgency_flag: boolean; finding: string }> {
  return request(`/api/cases/${caseId}/flag`, {
    method: 'POST',
    body: JSON.stringify({ finding, note: note ?? '' }),
  });
}

/** PATCH /api/cases/{id}/complete — mark a case as discharged */
export async function completeCase(caseId: string): Promise<{ status: string; case_id: string; discharged_at: string }> {
  return request(`/api/cases/${caseId}/complete`, { method: 'PATCH' });
}

/** GET /api/cases/{id}/similar — fetch similar cases via FAISS */
export async function fetchSimilarCases(
  caseId: string,
  topK: number = 3
): Promise<CaseSummary[]> {
  return request<CaseSummary[]>(`/api/cases/${caseId}/similar?top_k=${topK}`);
}

/** GET /api/cases/{id} — full multimodal detail (alias used by BeforeAfterTab) */
export function getCase(caseId: string): Promise<CaseDetail> {
  return request<CaseDetail>(`/api/cases/${caseId}`);
}

/** POST /api/cases/{id}/upload/cxr — upload a CXR image and trigger inference */
export async function uploadCXR(caseId: string, file: File): Promise<CaseDetail> {
  const form = new FormData();
  form.append('image', file);
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/cases/${caseId}/upload/cxr`, { method: 'POST', body: form, headers: authHeaders() });
  } catch {
    throw new ApiError('Unable to reach the backend server.', 0);
  }
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(body || `Request failed with status ${res.status}`, res.status);
  }
  return res.json() as Promise<CaseDetail>;
}

/** POST /api/cases/{id}/upload/ecg — update ECG data from JSON body */
export async function uploadECG(
  caseId: string,
  ecgData: Record<string, number | string>
): Promise<CaseDetail> {
  return request<CaseDetail>(`/api/cases/${caseId}/upload/ecg`, {
    method: 'POST',
    body: JSON.stringify(ecgData),
  });
}

/** POST /api/cases/{id}/upload/labs — parse and save lab results from file */
export async function uploadLabs(caseId: string, file: File): Promise<CaseDetail> {
  const form = new FormData();
  form.append('file', file);
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/cases/${caseId}/upload/labs`, { method: 'POST', body: form, headers: authHeaders() });
  } catch {
    throw new ApiError('Unable to reach the backend server.', 0);
  }
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new ApiError(body || `Request failed with status ${res.status}`, res.status);
  }
  return res.json() as Promise<CaseDetail>;
}
