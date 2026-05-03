/**
 * userRole.ts — dev-only role context (pre-auth).
 *
 * Lets us build role-gated UI without blocking on real Supabase Auth.
 * The four demo users below match the rows seeded by
 * `backend/seed_demo_users.py`.  When real auth lands (Phase 5), this
 * file is replaced by a Supabase session-derived hook.
 */
'use client';

import { useEffect, useState } from 'react';
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

function readRole(): UserRole {
  if (typeof window === 'undefined') return DEFAULT_ROLE;
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored && stored in DEMO_USERS) return stored as UserRole;
  return DEFAULT_ROLE;
}

/**
 * Returns the active role and a setter that persists to localStorage.
 * Hydrates after mount to avoid SSR/CSR mismatch (default role used pre-hydration).
 */
export function useUserRole(): {
  role: UserRole;
  user: typeof DEMO_USERS[UserRole];
  setRole: (r: UserRole) => void;
  hydrated: boolean;
} {
  const [role, setRoleState] = useState<UserRole>(DEFAULT_ROLE);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setRoleState(readRole());
    setHydrated(true);
  }, []);

  const setRole = (r: UserRole) => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(STORAGE_KEY, r);
    }
    setRoleState(r);
  };

  return { role, user: DEMO_USERS[role], setRole, hydrated };
}
