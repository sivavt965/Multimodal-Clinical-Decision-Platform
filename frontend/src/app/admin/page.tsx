'use client';

import React, { useEffect, useState, useCallback } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  ActivitySquare, ArrowLeft, Server, Database, Clock, FileStack,
  CheckCircle2, XCircle, RefreshCcw, Loader2, Wifi, WifiOff, Users, LogIn,
  ShieldAlert, ScrollText, Shield,
} from 'lucide-react';
import {
  listUsers, listAuditLog, createUser, updateUser,
  checkHealth, fetchCaseSummaries,
} from '@/lib/api';
import type { HealthResponse } from '@/lib/api';
import { useUserRole, ROLE_LABELS } from '@/lib/userRole';
import type { PlatformUser, AuditLogEntry, UserRole, CaseSummary } from '@/lib/types';

interface SessionRecord {
  id: string;
  userId: string;
  loginAt: string;
  durationMs: number | null;
}

function formatDuration(ms: number): string {
  const totalSecs = Math.floor(ms / 1000);
  const h = Math.floor(totalSecs / 3600);
  const m = Math.floor((totalSecs % 3600) / 60);
  const s = totalSecs % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}


export default function AdminPage() {
  const router = useRouter();
  const { role, hydrated, user } = useUserRole();

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Time-bearing state must be null on the server to avoid hydration mismatch;
  // initialised inside useEffect (client-only).
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [loginHistory, setLoginHistory] = useState<SessionRecord[]>([]);
  const [sessionStart, setSessionStart] = useState<Date | null>(null);
  const [now, setNow] = useState<Date | null>(null);

  // Phase 4d: User Management + Audit Log state
  const [users, setUsers] = useState<PlatformUser[]>([]);
  const [usersError, setUsersError] = useState<string | null>(null);
  const [auditError, setAuditError] = useState<string | null>(null);
  const [auditEntries, setAuditEntries] = useState<AuditLogEntry[]>([]);
  const [auditTotal, setAuditTotal] = useState(0);
  const [auditFilter, setAuditFilter] = useState<{ action: string; userId: string }>({ action: '', userId: '' });
  const [showAddUser, setShowAddUser] = useState(false);
  const [savingUserId, setSavingUserId] = useState<string | null>(null);
  const [addForm, setAddForm] = useState<{ email: string; full_name: string; role: UserRole }>({
    email: '', full_name: '', role: 'ward_doctor',
  });
  const [addError, setAddError] = useState<string | null>(null);

  const refreshUsers = async () => {
    try {
      setUsersError(null);
      setUsers(await listUsers());
    } catch (err: any) {
      setUsersError(err?.message || 'Failed to load users');
    }
  };

  const handleAddUser = async () => {
    setAddError(null);
    if (!addForm.email || !addForm.full_name) {
      setAddError('Email and name are required.');
      return;
    }
    try {
      await createUser(addForm);
      setShowAddUser(false);
      setAddForm({ email: '', full_name: '', role: 'ward_doctor' });
      await refreshUsers();
    } catch (err: any) {
      setAddError(err?.message || 'Failed to create user');
    }
  };

  const handleChangeRole = async (userId: string, newRole: UserRole) => {
    setSavingUserId(userId);
    try {
      await updateUser(userId, { role: newRole });
      await refreshUsers();
    } catch {} finally { setSavingUserId(null); }
  };

  const handleToggleStatus = async (u: PlatformUser) => {
    setSavingUserId(u.id);
    try {
      await updateUser(u.id, { status: u.status === 'active' ? 'inactive' : 'active' });
      await refreshUsers();
    } catch {} finally { setSavingUserId(null); }
  };

  // Gate the page: only system_admin may view. Other roles get bounced to /dashboard.
  useEffect(() => {
    if (!hydrated) return;
    if (role !== 'system_admin') router.replace('/dashboard');
  }, [hydrated, role, router]);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [healthData, casesData] = await Promise.all([
        checkHealth(),
        fetchCaseSummaries().catch(() => [] as CaseSummary[]),
      ]);
      setHealth(healthData);
      setCases(casesData);
      setLastRefresh(new Date());
    } catch (err: any) {
      setError(err.message || 'Failed to connect to backend');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Auto-refresh every 30s
    return () => clearInterval(interval);
  }, [fetchData]);

  // User Management — load users on mount
  useEffect(() => {
    listUsers().then(setUsers).catch((err: any) => {
      setUsersError(err?.message || 'Failed to load users');
    });
  }, []);

  // Audit Log — refetch when filter changes
  useEffect(() => {
    setAuditError(null);
    listAuditLog({
      limit: 100,
      action: auditFilter.action || undefined,
      userId: auditFilter.userId || undefined,
    })
      .then((r) => { setAuditEntries(r.items); setAuditTotal(r.total); })
      .catch((err: any) => {
        setAuditError(err?.message || 'Failed to load audit log');
      });
  }, [auditFilter]);

  // Session tracking
  useEffect(() => {
    const stored = sessionStorage.getItem('session_start');
    const sessionId = sessionStorage.getItem('session_id');

    if (!stored || !sessionId) {
      const now = new Date();
      const newId = `sess_${Date.now().toString(36)}`;
      sessionStorage.setItem('session_start', now.toISOString());
      sessionStorage.setItem('session_id', newId);
      setSessionStart(now);

      // Append to login history in localStorage
      const historyRaw = localStorage.getItem('login_history');
      const history: SessionRecord[] = historyRaw ? JSON.parse(historyRaw) : [];
      const newRecord: SessionRecord = {
        id: newId,
        userId: user.id,
        loginAt: now.toISOString(),
        durationMs: null,
      };
      const updated = [newRecord, ...history].slice(0, 20);
      localStorage.setItem('login_history', JSON.stringify(updated));
      setLoginHistory(updated);
    } else {
      setSessionStart(new Date(stored));
      const historyRaw = localStorage.getItem('login_history');
      if (historyRaw) setLoginHistory(JSON.parse(historyRaw));
    }
  }, []);

  // Live clock tick for session duration (client-only)
  useEffect(() => {
    setNow(new Date());
    const ticker = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(ticker);
  }, []);

  const isOnline = health?.status === 'ok';
  const dbOk = health?.db_status?.startsWith('ok') ?? false;

  const metrics = [
    {
      title: 'Total Cases',
      value: health ? String(health.cases_in_db) : '—',
      icon: FileStack,
      color: 'text-blue-600',
      bg: 'bg-blue-100',
    },
    {
      title: 'Open Consultations',
      value: String(cases.filter(c => c.consultation_open).length),
      icon: ActivitySquare,
      color: 'text-emerald-600',
      bg: 'bg-emerald-100',
    },
    {
      title: 'FAISS Index',
      value: health ? `${health.faiss_index_size} vec` : '—',
      icon: Database,
      color: 'text-purple-600',
      bg: 'bg-purple-100',
    },
    {
      title: 'DB Status',
      value: dbOk ? 'Connected' : 'Error',
      icon: Clock,
      color: dbOk ? 'text-emerald-600' : 'text-red-600',
      bg: dbOk ? 'bg-emerald-100' : 'bg-red-100',
    },
  ];

  // Build logs from recent cases (most recent 10)
  const recentLogs = cases.slice(0, 10).map((c, i) => {
    const date = new Date(c.admitted_at);
    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return {
      id: c.case_id.substring(0, 8),
      time: timeStr,
      type: c.top_finding_label ? 'Multimodal Inference' : 'Case Ingestion',
      user: c.patient_name || 'Unknown',
      status: c.top_finding_label ? 'Processed' : 'Pending',
      risk: c.phase_a_risk_level || '—',
    };
  });

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Top Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-4">
          <Link href="/dashboard" className="p-2 hover:bg-slate-100 rounded-full transition-colors">
            <ArrowLeft className="w-5 h-5 text-slate-600" />
          </Link>
          <div className="flex items-center gap-2">
            <Server className="w-6 h-6 text-slate-800" />
            <h1 className="text-xl font-bold text-slate-900">System Health Dashboard</h1>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={fetchData}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 border border-slate-200 rounded-lg hover:bg-slate-200 text-xs font-semibold text-slate-600 transition-colors disabled:opacity-50"
          >
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCcw className="w-3.5 h-3.5" />}
            Refresh
          </button>
          <div className={`flex items-center gap-2 px-3 py-1.5 border rounded-full ${isOnline ? 'bg-emerald-50 border-emerald-200' : 'bg-red-50 border-red-200'}`}>
            {isOnline ? (
              <>
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
                </span>
                <span className="text-xs font-bold text-emerald-700 uppercase tracking-wide">Online</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5 text-red-500" />
                <span className="text-xs font-bold text-red-700 uppercase tracking-wide">
                  {error ? 'Offline' : 'Checking...'}
                </span>
              </>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 p-6 max-w-[1200px] mx-auto w-full flex flex-col gap-8">

        {/* Error Banner */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-center gap-3">
            <XCircle className="w-5 h-5 text-red-500 shrink-0" />
            <div>
              <p className="text-sm font-semibold text-red-800">Backend Connection Failed</p>
              <p className="text-xs text-red-600">{error}</p>
            </div>
          </div>
        )}

        {/* Metrics Grid */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-bold text-slate-500 uppercase tracking-widest">Live Metrics</h2>
            <span className="text-[10px] text-slate-400 font-mono">
              Last refresh: {lastRefresh ? lastRefresh.toLocaleTimeString() : '—'}
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {metrics.map((metric, i) => {
              const Icon = metric.icon;
              return (
                <div
                  key={metric.title}
                  className="group bg-white border border-slate-100 rounded-xl p-5 shadow-card hover:shadow-card-hover hover:-translate-y-0.5 hover:border-slate-200 transition-all duration-200 flex items-center gap-4 opacity-0 animate-fadeInUp"
                  style={{ animationDelay: `${i * 60}ms`, animationFillMode: 'both' }}
                >
                  <div className={`p-3 rounded-xl ${metric.bg} group-hover:scale-105 transition-transform duration-200`}>
                    <Icon className={`w-6 h-6 ${metric.color}`} />
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-slate-500">{metric.title}</p>
                    <p className="text-2xl font-bold text-slate-900 tabular-nums leading-tight mt-0.5">
                      {loading && !health ? <Loader2 className="w-5 h-5 animate-spin text-slate-400" /> : metric.value}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Logs Table */}
        <section className="flex-1">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
              <FileStack className="w-4 h-4 text-slate-400" />
              Recent Cases
            </h2>
            <span className="text-[10px] font-mono text-slate-400">{recentLogs.length} shown</span>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl shadow-card overflow-hidden">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200">
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Case ID</th>
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Admitted</th>
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Patient</th>
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Phase A Risk</th>
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Top Finding</th>
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {recentLogs.length === 0 && !loading ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center text-sm text-slate-400">
                      <FileStack className="w-8 h-8 text-slate-200 mx-auto mb-2" />
                      No cases found. Register a case via the Clinical Ingestion Wizard.
                    </td>
                  </tr>
                ) : (
                  recentLogs.map((log, idx) => (
                    <tr
                      key={log.id}
                      className="group hover:bg-blue-50/40 transition-colors duration-150 opacity-0 animate-fadeInUp"
                      style={{ animationDelay: `${idx * 35}ms`, animationFillMode: 'both' }}
                    >
                      <td className="px-6 py-3.5 text-sm font-mono text-slate-500 group-hover:text-blue-700 transition-colors">{log.id}</td>
                      <td className="px-6 py-3.5 text-sm text-slate-600">{log.time}</td>
                      <td className="px-6 py-3.5 text-sm font-semibold text-slate-900">{log.user}</td>
                      <td className="px-6 py-3.5">
                        <span className={`text-[10px] font-bold px-2 py-1 rounded-md border tracking-wider uppercase ${
                          log.risk === 'High' ? 'bg-red-50 text-red-700 border-red-200' :
                          log.risk === 'Moderate' ? 'bg-amber-50 text-amber-700 border-amber-200' :
                          log.risk === 'Low' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                          'bg-slate-50 text-slate-500 border-slate-200'
                        }`}>
                          {log.risk}
                        </span>
                      </td>
                      <td className="px-6 py-3.5 text-sm text-slate-600">{log.type}</td>
                      <td className="px-6 py-3.5">
                        {log.status === 'Processed' ? (
                          <div className="inline-flex items-center gap-1.5 text-emerald-700 bg-emerald-50 px-2 py-1 rounded-md w-fit border border-emerald-200">
                            <CheckCircle2 className="w-3.5 h-3.5" />
                            <span className="text-[10px] font-bold uppercase tracking-wider">Processed</span>
                          </div>
                        ) : (
                          <div className="inline-flex items-center gap-1.5 text-amber-700 bg-amber-50 px-2 py-1 rounded-md w-fit border border-amber-200">
                            <Clock className="w-3.5 h-3.5" />
                            <span className="text-[10px] font-bold uppercase tracking-wider">Pending</span>
                          </div>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* Login Activity */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
              <LogIn className="w-4 h-4 text-slate-400" />
              Login Activity
            </h2>
            {sessionStart && now && (
              <div className="flex items-center gap-2 text-xs text-slate-500 font-mono bg-white border border-emerald-200 px-3 py-1 rounded-full shadow-sm">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span className="text-emerald-700 font-semibold">Active:</span> {formatDuration(now.getTime() - sessionStart.getTime())}
              </div>
            )}
          </div>

          {/* Current Session Card */}
          {sessionStart && (
            <div className="relative bg-white border border-slate-200 rounded-xl p-5 mb-4 shadow-card flex items-center gap-4 overflow-hidden">
              <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-blue-500 to-emerald-500" />
              <div className="p-3 rounded-xl bg-blue-100 shrink-0">
                <Users className="w-6 h-6 text-blue-600" />
              </div>
              <div className="flex-1 grid grid-cols-3 gap-4">
                <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-0.5">User ID</p>
                  <p className="text-sm font-bold text-slate-900 font-mono">{user.id}</p>
                  <p className="text-xs text-slate-500">{user.full_name}</p>
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-0.5">Login Time</p>
                  <p className="text-sm font-semibold text-slate-900">{sessionStart.toLocaleTimeString()}</p>
                  <p className="text-xs text-slate-500">{sessionStart.toLocaleDateString()}</p>
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-0.5">Session Duration</p>
                  <p className="text-sm font-bold text-emerald-700 font-mono tabular-nums">{now ? formatDuration(now.getTime() - sessionStart.getTime()) : '—'}</p>
                  <p className="text-xs text-slate-500 flex items-center gap-1">
                    <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulseSoft" />
                    Active now
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Login History Table */}
          <div className="bg-white border border-slate-200 rounded-xl shadow-card overflow-hidden">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200">
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Session ID</th>
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">User</th>
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Login Time</th>
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Duration</th>
                  <th className="px-6 py-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {loginHistory.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-6 py-8 text-center text-sm text-slate-400">
                      No login history found.
                    </td>
                  </tr>
                ) : loginHistory.map((record, idx) => {
                  const isCurrentSession = record.id === sessionStorage.getItem('session_id');
                  const loginDate = new Date(record.loginAt);
                  const duration = isCurrentSession
                    ? formatDuration(now.getTime() - loginDate.getTime())
                    : record.durationMs !== null
                      ? formatDuration(record.durationMs)
                      : '—';
                  return (
                    <tr
                      key={record.id}
                      className={`opacity-0 animate-fadeInUp transition-colors duration-150 ${
                        isCurrentSession ? 'bg-emerald-50/40 hover:bg-emerald-50/70' : 'hover:bg-blue-50/40'
                      }`}
                      style={{ animationDelay: `${idx * 30}ms`, animationFillMode: 'both' }}
                    >
                      <td className="px-6 py-3 text-xs font-mono text-slate-500">{record.id}</td>
                      <td className="px-6 py-3 text-sm font-semibold text-slate-800">{user.full_name}</td>
                      <td className="px-6 py-3 text-sm text-slate-600">
                        {loginDate.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        <span className="ml-1.5 text-xs text-slate-400">{loginDate.toLocaleDateString()}</span>
                      </td>
                      <td className="px-6 py-3 text-sm font-mono text-slate-700 tabular-nums">{duration}</td>
                      <td className="px-6 py-3">
                        {isCurrentSession ? (
                          <div className="inline-flex items-center gap-1.5 text-emerald-700 bg-emerald-50 px-2 py-1 rounded-md w-fit border border-emerald-200">
                            <span className="relative flex h-1.5 w-1.5">
                              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
                            </span>
                            <span className="text-[10px] font-bold uppercase tracking-wider">Active</span>
                          </div>
                        ) : (
                          <div className="inline-flex items-center gap-1.5 text-slate-500 bg-slate-50 px-2 py-1 rounded-md w-fit border border-slate-200">
                            <LogIn className="w-3 h-3" />
                            <span className="text-[10px] font-bold uppercase tracking-wider">Ended</span>
                          </div>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* User Management */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
              <Shield className="w-4 h-4 text-slate-400" />
              User Management
            </h2>
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-400 tabular-nums">{users.length} users</span>
              <button
                onClick={() => { setAddError(null); setShowAddUser(true); }}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold rounded-lg shadow-sm transition-all duration-150"
              >
                + Add User
              </button>
            </div>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl shadow-card overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[11px] font-semibold uppercase tracking-wider text-slate-400 bg-slate-50">
                  <th className="px-5 py-2.5">User</th>
                  <th className="px-5 py-2.5">Role</th>
                  <th className="px-5 py-2.5">Status</th>
                  <th className="px-5 py-2.5">Last active</th>
                  <th className="px-5 py-2.5 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u, i) => {
                  const saving = savingUserId === u.id;
                  return (
                  <tr key={u.id} className={`border-t border-slate-100 ${i % 2 === 1 ? 'bg-slate-50/40' : ''}`}>
                    <td className="px-5 py-3">
                      <p className="font-medium text-slate-900">{u.full_name}</p>
                      <p className="text-xs text-slate-500 font-mono">{u.email}</p>
                    </td>
                    <td className="px-5 py-3">
                      <select
                        value={u.role}
                        disabled={saving}
                        onChange={(e) => handleChangeRole(u.id, e.target.value as UserRole)}
                        className="text-xs px-2 py-1 border border-slate-200 rounded-md bg-white disabled:opacity-50"
                      >
                        {(['radiologist','ward_doctor','clinical_admin','system_admin'] as UserRole[]).map((r) => (
                          <option key={r} value={r}>{ROLE_LABELS[r]}</option>
                        ))}
                      </select>
                    </td>
                    <td className="px-5 py-3">
                      <span className={`inline-flex items-center gap-1.5 text-xs font-medium ${u.status === 'active' ? 'text-emerald-700' : 'text-slate-500'}`}>
                        <span className={`w-1.5 h-1.5 rounded-full ${u.status === 'active' ? 'bg-emerald-500' : 'bg-slate-400'}`} />
                        {u.status}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-xs text-slate-500 tabular-nums">
                      {u.last_active_at ? new Date(u.last_active_at).toLocaleString() : '—'}
                    </td>
                    <td className="px-5 py-3 text-right">
                      <button
                        onClick={() => handleToggleStatus(u)}
                        disabled={saving}
                        className={`text-xs font-medium px-2.5 py-1 rounded-md border transition-colors disabled:opacity-50 ${
                          u.status === 'active'
                            ? 'border-slate-200 text-slate-600 hover:bg-slate-50'
                            : 'border-emerald-200 text-emerald-700 hover:bg-emerald-50'
                        }`}
                      >
                        {u.status === 'active' ? 'Deactivate' : 'Reactivate'}
                      </button>
                    </td>
                  </tr>
                  );
                })}
                {users.length === 0 && (
                  <tr><td colSpan={5} className="px-5 py-10 text-center text-sm text-slate-400">No users registered.</td></tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Add User dialog */}
          {showAddUser && (
            <div
              className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fadeIn"
              onClick={() => setShowAddUser(false)}
            >
              <div
                className="bg-white rounded-2xl shadow-xl w-full max-w-md p-6 animate-scaleIn"
                onClick={(e) => e.stopPropagation()}
              >
                <h3 className="text-lg font-semibold text-slate-900 mb-1">Add new user</h3>
                <p className="text-sm text-slate-500 mb-5">They&apos;ll be created in <span className="font-mono">active</span> status.</p>

                <div className="space-y-3">
                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">Email</label>
                    <input
                      type="email"
                      value={addForm.email}
                      onChange={(e) => setAddForm((f) => ({ ...f, email: e.target.value }))}
                      placeholder="dr.example@hospital.org"
                      className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">Full name</label>
                    <input
                      value={addForm.full_name}
                      onChange={(e) => setAddForm((f) => ({ ...f, full_name: e.target.value }))}
                      placeholder="Dr. Example Name"
                      className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">Role</label>
                    <select
                      value={addForm.role}
                      onChange={(e) => setAddForm((f) => ({ ...f, role: e.target.value as UserRole }))}
                      className="w-full px-3 py-2 text-sm border border-slate-200 rounded-lg bg-white"
                    >
                      {(['radiologist','ward_doctor','clinical_admin','system_admin'] as UserRole[]).map((r) => (
                        <option key={r} value={r}>{ROLE_LABELS[r]}</option>
                      ))}
                    </select>
                  </div>
                </div>

                {addError && (
                  <p className="mt-3 text-sm text-red-600">{addError}</p>
                )}

                <div className="mt-6 flex items-center justify-end gap-2">
                  <button
                    onClick={() => setShowAddUser(false)}
                    className="px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleAddUser}
                    className="px-4 py-2 text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg shadow-sm transition-colors"
                  >
                    Create user
                  </button>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Audit Log */}
        <section>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
              <ScrollText className="w-4 h-4 text-slate-400" />
              Audit Log
            </h2>
            <div className="flex items-center gap-2">
              <select
                value={auditFilter.action}
                onChange={(e) => setAuditFilter((f) => ({ ...f, action: e.target.value }))}
                className="text-xs px-2 py-1 border border-slate-200 rounded-md bg-white"
              >
                <option value="">All actions</option>
                {Array.from(new Set(auditEntries.map((e) => e.action))).map((a) => (
                  <option key={a} value={a}>{a}</option>
                ))}
              </select>
              <select
                value={auditFilter.userId}
                onChange={(e) => setAuditFilter((f) => ({ ...f, userId: e.target.value }))}
                className="text-xs px-2 py-1 border border-slate-200 rounded-md bg-white"
              >
                <option value="">All users</option>
                {users.map((u) => <option key={u.id} value={u.id}>{u.full_name}</option>)}
              </select>
              <span className="text-xs text-slate-400 tabular-nums">{auditEntries.length}/{auditTotal}</span>
            </div>
          </div>
          <div className="bg-white border border-slate-200 rounded-xl shadow-card overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[11px] font-semibold uppercase tracking-wider text-slate-400 bg-slate-50">
                  <th className="px-5 py-2.5">Time</th>
                  <th className="px-5 py-2.5">User</th>
                  <th className="px-5 py-2.5">Action</th>
                  <th className="px-5 py-2.5">Target</th>
                  <th className="px-5 py-2.5">Detail</th>
                </tr>
              </thead>
              <tbody>
                {auditEntries.map((e, i) => {
                  const u = users.find((x) => x.id === e.user_id);
                  return (
                    <tr key={e.id} className={`border-t border-slate-100 ${i % 2 === 1 ? 'bg-slate-50/40' : ''}`}>
                      <td className="px-5 py-3 text-xs text-slate-500 font-mono tabular-nums whitespace-nowrap">
                        {new Date(e.created_at).toLocaleTimeString()}
                        <span className="block text-[10px] text-slate-400">{new Date(e.created_at).toLocaleDateString()}</span>
                      </td>
                      <td className="px-5 py-3">
                        <p className="text-xs font-medium text-slate-900">{u?.full_name ?? '—'}</p>
                        <p className="text-[10px] text-slate-400">{e.user_role ?? ''}</p>
                      </td>
                      <td className="px-5 py-3">
                        <span className="inline-flex px-2 py-0.5 rounded text-[11px] font-mono bg-slate-100 text-slate-700 border border-slate-200">
                          {e.action}
                        </span>
                      </td>
                      <td className="px-5 py-3 text-xs text-slate-500 font-mono tabular-nums">
                        {e.target_type ?? '—'}{e.target_id ? ` · ${e.target_id.slice(0, 8)}` : ''}
                      </td>
                      <td className="px-5 py-3 text-xs text-slate-500 font-mono">
                        {Object.entries(e.metadata)
                          .filter(([k]) => k !== 'seed')
                          .map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`)
                          .join('  ')}
                      </td>
                    </tr>
                  );
                })}
                {auditEntries.length === 0 && (
                  <tr><td colSpan={5} className="px-5 py-10 text-center text-sm text-slate-400">No audit entries yet.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* Raw Health JSON (Debug) */}
        {health && (
          <section>
            <h2 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
              <Server className="w-4 h-4 text-slate-400" />
              Raw Health Response
              <span className="ml-1 text-[10px] font-mono text-slate-400 normal-case tracking-normal">debug</span>
            </h2>
            <div className="relative bg-slate-900 rounded-xl border border-slate-800 shadow-card overflow-hidden">
              <div className="flex items-center gap-1.5 px-4 py-2 border-b border-slate-800 bg-slate-950/50">
                <span className="w-2.5 h-2.5 rounded-full bg-red-500/70" />
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500/70" />
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/70" />
                <span className="ml-2 text-[10px] font-mono text-slate-500">GET /api/health</span>
              </div>
              <pre className="text-emerald-300 p-4 text-xs font-mono overflow-x-auto leading-relaxed">
                {JSON.stringify(health, null, 2)}
              </pre>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
