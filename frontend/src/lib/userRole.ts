/**
 * userRole.ts — Role context, Phase 5 edition.
 *
 * Priority:
 *  1. Real Supabase session (useAuth) → role from /api/me lookup
 *  2. Dev localStorage shim (cdss_dev_role) — active when no real session
 *
 * The `DEMO_USERS` roster and `RoleSwitcher` component are kept for the
 * dev shim path so local development still works without a live auth session.
 */
'use client';

import { useEffect, useState, useSyncExternalStore } from 'react';
import { useAuth } from './auth';
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

// ── Dev shim external store ──────────────────────────────────────────────────
const listeners = new Set<() => void>();

function readDevRole(): UserRole {
  if (typeof window === 'undefined') return DEFAULT_ROLE;
  const v = window.localStorage.getItem(STORAGE_KEY);
  return v && v in DEMO_USERS ? (v as UserRole) : DEFAULT_ROLE;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
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

// ── Combined hook ─────────────────────────────────────────────────────────────

/**
 * Returns the active role and user, synced across all components.
 *
 * If a real Supabase session is active, role and user come from the
 * authenticated profile. Otherwise falls back to the dev localStorage shim.
 */
export function useUserRole(): {
  role: UserRole;
  user: Pick<PlatformUser, 'id' | 'email' | 'full_name' | 'role'>;
  setRole: (r: UserRole) => void;
  hydrated: boolean;
  isRealSession: boolean;
} {
  const { session, userProfile, loading } = useAuth();
  const devRole = useSyncExternalStore(subscribe, readDevRole, getServerSnapshot);
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => { setHydrated(true); }, []);

  const isRealSession = !loading && !!session && !!userProfile;

  if (isRealSession && userProfile) {
    return {
      role: userProfile.role,
      user: {
        id:        userProfile.id,
        email:     userProfile.email,
        full_name: userProfile.full_name,
        role:      userProfile.role,
      },
      setRole: setRoleGlobal, // no-op in real auth (role is from DB)
      hydrated: true,
      isRealSession: true,
    };
  }

  return {
    role:          devRole,
    user:          DEMO_USERS[devRole],
    setRole:       setRoleGlobal,
    hydrated,
    isRealSession: false,
  };
}
