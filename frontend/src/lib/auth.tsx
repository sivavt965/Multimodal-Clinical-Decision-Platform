'use client';

/**
 * auth.tsx — Supabase Auth context for Phase 5.
 *
 * Wraps the app in an AuthProvider that tracks the Supabase session and
 * resolves the user's application role via GET /api/me. Components call
 * useAuth() to get { session, userProfile, loading }.
 *
 * Design: role resolution is async (one /api/me fetch after sign-in), so
 * there's a brief window where session exists but userProfile is null —
 * callers should gate on `loading` or check `userProfile` explicitly.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from 'react';
import type { Session } from '@supabase/supabase-js';
import { supabase } from './supabase';
import type { UserRole } from './types';

export interface UserProfile {
  id: string;
  email: string;
  full_name: string;
  role: UserRole;
}

interface AuthState {
  session: Session | null;
  userProfile: UserProfile | null;
  loading: boolean;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthState>({
  session: null,
  userProfile: null,
  loading: true,
  signOut: async () => {},
});

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

async function fetchUserProfile(token: string): Promise<UserProfile | null> {
  try {
    const res = await fetch(`${API_BASE}/api/me`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) return null;
    return res.json() as Promise<UserProfile>;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  // Track the last resolved token so we don't re-fetch on unrelated re-renders.
  const resolvedToken = useRef<string | null>(null);

  const resolveProfile = useCallback(async (s: Session | null) => {
    if (!s) {
      setUserProfile(null);
      resolvedToken.current = null;
      return;
    }
    const token = s.access_token;
    if (token === resolvedToken.current) return;
    resolvedToken.current = token;
    const profile = await fetchUserProfile(token);
    setUserProfile(profile);
  }, []);

  useEffect(() => {
    // Bootstrap: grab the current session (cached by Supabase SDK).
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      resolveProfile(data.session).finally(() => setLoading(false));
    });

    // Subscribe to auth state changes (login, logout, token refresh).
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, newSession) => {
        setSession(newSession);
        resolveProfile(newSession);
        setLoading(false);
      },
    );

    return () => subscription.unsubscribe();
  }, [resolveProfile]);

  const signOut = useCallback(async () => {
    await supabase.auth.signOut();
    setSession(null);
    setUserProfile(null);
  }, []);

  return (
    <AuthContext.Provider value={{ session, userProfile, loading, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  return useContext(AuthContext);
}
