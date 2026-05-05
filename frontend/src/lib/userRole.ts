/**
 * userRole.ts — dev-only role context (pre-auth).
 *
 * Lets us build role-gated UI without blocking on real Supabase Auth.
 * The four demo users below match the rows seeded by
 * `backend/seed_demo_users.py`.  When real auth lands (Phase 5), this
 * file is replaced by a Supabase session-derived hook.
 *
 * Internals: a tiny external store + useSyncExternalStore so that every
 * consumer of useUserRole() re-renders when ANY component calls setRole().
 * Without this, each useState was local and switching roles in the header
 * left other components stale until a manual refresh.
 */
'use client';

import { useEffect, useState, useSyncExternalStore } from 'react';
import type { UserRole, PlatformUser } from './types';

const STORAGE_KEY = 'cdss_dev_role';
const DEFAULT_ROLE: UserRole = 'ward_doctor';

/** Demo user roster — UUIDs come from `seed_demo_users.py` upserts. */
export const DEMO_USERS: Record<UserRole, Pick<PlatformUser, 'id' | 'email' | 'full_name' | 'role'>> = {
  radiologist: {
    id: '4f9b9bc8-bfd7-4b3f-924a-0dae0c882f90',
    email: 'dr.smith@hospital.org',
    full_name: 'Dr. Alice Smith',
    role: 'radiologist',
  },
  ward_doctor: {
    id: 'c29a01e9-f3e6-4a1e-8aff-35a11d49b57c',
    email: 'dr.johnson@hospital.org',
    full_name: 'Dr. Ben Johnson',
    role: 'ward_doctor',
  },
  clinical_admin: {
    id: '5a351665-167b-429f-b6c9-635237995e0f',
    email: 'sarah.lee@hospital.org',
    full_name: 'Sarah Lee',
    role: 'clinical_admin',
  },
  system_admin: {
    id: '586531d7-daf0-48f5-80e9-dbbfdfcfcc4b',
    email: 'ops@hospital.org',
    full_name: 'System Operator',
    role: 'system_admin',
  },
};

export const ROLE_LABELS: Record<UserRole, string> = {
  radiologist:    'Radiologist',
  ward_doctor:    'Ward Doctor',
  clinical_admin: 'Clinical Admin',
  system_admin:   'System Admin',
};

// ── External store wiring ────────────────────────────────────────────────
const listeners = new Set<() => void>();

function readRole(): UserRole {
  if (typeof window === 'undefined') return DEFAULT_ROLE;
  const v = window.localStorage.getItem(STORAGE_KEY);
  return v && v in DEMO_USERS ? (v as UserRole) : DEFAULT_ROLE;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  // Also listen for storage events from other tabs.
  if (typeof window !== 'undefined') {
    window.addEventListener('storage', listener);
  }
  return () => {
    listeners.delete(listener);
    if (typeof window !== 'undefined') {
      window.removeEventListener('storage', listener);
    }
  };
}

function setRoleGlobal(r: UserRole): void {
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(STORAGE_KEY, r);
  }
  listeners.forEach((l) => l());
}

const getServerSnapshot = (): UserRole => DEFAULT_ROLE;

/**
 * Returns the active role (synced across all components) and a global
 * setter. Hydrated flag stays around so consumers can skip render until
 * client-side mount when they need to avoid a flash of the default role.
 */
export function useUserRole(): {
  role: UserRole;
  user: typeof DEMO_USERS[UserRole];
  setRole: (r: UserRole) => void;
  hydrated: boolean;
} {
  const role = useSyncExternalStore(subscribe, readRole, getServerSnapshot);
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => { setHydrated(true); }, []);
  return { role, user: DEMO_USERS[role], setRole: setRoleGlobal, hydrated };
}
