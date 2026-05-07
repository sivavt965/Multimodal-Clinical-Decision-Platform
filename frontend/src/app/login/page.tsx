'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { ShieldCheck, Lock, User, ActivitySquare, AlertCircle } from 'lucide-react';
import { supabase } from '@/lib/supabase';
import { useAuth } from '@/lib/auth';

export default function LoginPage() {
  const router = useRouter();
  const { session, loading } = useAuth();
  const [email, setEmail] = useState('dr.smith@hospital.org');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Already signed in — go straight to dashboard.
  useEffect(() => {
    if (!loading && session) {
      router.replace('/dashboard');
    }
  }, [session, loading, router]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    const { error: authError } = await supabase.auth.signInWithPassword({
      email: email.trim(),
      password,
    });

    if (authError) {
      setError(authError.message);
      setIsLoading(false);
      return;
    }

    router.replace('/dashboard');
  };

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-100 flex flex-col justify-center items-center p-4 overflow-hidden">

      {/* Decorative background blobs */}
      <div className="pointer-events-none absolute -top-32 -left-32 w-96 h-96 bg-blue-200/30 rounded-full blur-3xl" />
      <div className="pointer-events-none absolute -bottom-32 -right-32 w-96 h-96 bg-emerald-200/30 rounded-full blur-3xl" />

      {/* Brand Header */}
      <div className="relative mb-8 flex flex-col items-center animate-fadeInUp">
        <div className="bg-gradient-to-br from-blue-600 to-blue-700 p-3.5 rounded-2xl shadow-lg shadow-blue-500/30 mb-4 hover:scale-105 transition-transform duration-200">
          <ActivitySquare className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-2xl font-bold text-slate-900 tracking-tight">Clinical Decision Support</h1>
        <p className="text-slate-500 text-sm mt-1">Symile-MIMIC Multimodal Platform</p>
      </div>

      {/* Login Card */}
      <div
        className="relative bg-white border border-slate-200 rounded-2xl shadow-2xl shadow-slate-300/40 w-full max-w-md overflow-hidden opacity-0 animate-fadeInUp"
        style={{ animationDelay: '120ms', animationFillMode: 'both' }}
      >
        <div className="p-8">
          <h2 className="text-xl font-bold text-slate-800 mb-1">Staff Portal Login</h2>
          <p className="text-xs text-slate-500 mb-6">Sign in with your hospital credentials to continue.</p>

          {error && (
            <div className="mb-4 flex items-start gap-2 bg-red-50 border border-red-200 rounded-lg px-3 py-2.5 text-sm text-red-700">
              <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleLogin} className="space-y-5">
            <div>
              <label className="block text-xs font-bold text-slate-600 mb-1.5 uppercase tracking-wider">Staff ID / Email</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <User className="h-5 w-5 text-slate-400 group-focus-within:text-blue-600 transition-colors" />
                </div>
                <input
                  type="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2.5 border border-slate-300 rounded-lg text-slate-900 focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500 transition-all sm:text-sm"
                  placeholder="you@hospital.org"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-600 mb-1.5 uppercase tracking-wider">Password</label>
              <div className="relative group">
                <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                  <Lock className="h-5 w-5 text-slate-400 group-focus-within:text-blue-600 transition-colors" />
                </div>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="block w-full pl-10 pr-3 py-2.5 border border-slate-300 rounded-lg text-slate-900 focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500 transition-all sm:text-sm"
                  placeholder="••••••••"
                />
              </div>
            </div>

            <div className="pt-2">
              <button
                type="submit"
                disabled={isLoading}
                className="w-full flex justify-center items-center gap-2 py-2.5 px-4 rounded-lg shadow-md shadow-blue-500/20 text-sm font-bold text-white bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 hover:shadow-lg hover:shadow-blue-500/30 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-600 transition-all duration-200 disabled:opacity-70 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <>
                    <span className="inline-block w-4 h-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                    Authenticating...
                  </>
                ) : 'Sign In'}
              </button>
            </div>
          </form>
        </div>

        {/* Secure Badge Footer */}
        <div className="bg-slate-50 px-8 py-4 border-t border-slate-100 flex items-center justify-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-600" />
          <span className="text-[11px] font-bold text-emerald-800 tracking-wider uppercase">256-Bit Secure Connection</span>
        </div>
      </div>

      {/* Footer Links */}
      <div
        className="relative mt-8 text-center text-xs text-slate-500 opacity-0 animate-fadeInUp"
        style={{ animationDelay: '240ms', animationFillMode: 'both' }}
      >
        <p>Unauthorized access is strictly prohibited.</p>
        <p className="mt-1">For access issues, contact IT Support at <span className="font-semibold text-slate-700">ext. 4992</span>.</p>
      </div>

    </div>
  );
}
