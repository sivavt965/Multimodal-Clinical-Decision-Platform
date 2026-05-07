'use client';

import React, { useEffect, useRef, useState } from 'react';
import { ChevronDown, UserCheck, Stethoscope, ClipboardList, ShieldAlert } from 'lucide-react';
import { useUserRole, ROLE_LABELS, DEMO_USERS } from '@/lib/userRole';
import type { UserRole } from '@/lib/types';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

const ROLE_ICON: Record<UserRole, React.ComponentType<{ className?: string }>> = {
  radiologist:    Stethoscope,
  ward_doctor:    UserCheck,
  clinical_admin: ClipboardList,
  system_admin:   ShieldAlert,
};

const ROLE_COLOR: Record<UserRole, string> = {
  radiologist:    'text-blue-600 bg-blue-50 border-blue-200',
  ward_doctor:    'text-emerald-600 bg-emerald-50 border-emerald-200',
  clinical_admin: 'text-violet-600 bg-violet-50 border-violet-200',
  system_admin:   'text-rose-600 bg-rose-50 border-rose-200',
};

const ROLES: UserRole[] = ['ward_doctor', 'radiologist', 'clinical_admin', 'system_admin'];

/** Static chip shown when the user is authenticated via real Supabase session. */
function UserChip({ role, name }: { role: UserRole; name: string }) {
  const Icon = ROLE_ICON[role];
  return (
    <div className={cn(
      "px-3 py-1.5 rounded-lg flex items-center gap-2 border text-sm font-medium select-none",
      ROLE_COLOR[role],
    )}>
      <Icon className="h-4 w-4 shrink-0" />
      <span className="hidden sm:inline">{name}</span>
      <span className="hidden sm:inline text-[11px] font-normal opacity-60">· {ROLE_LABELS[role]}</span>
    </div>
  );
}

/** Dev-only dropdown — only rendered when no real session is active. */
export function RoleSwitcher() {
  const { role, user, setRole, hydrated, isRealSession } = useUserRole();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    return () => document.removeEventListener('mousedown', onClick);
  }, [open]);

  if (!hydrated) {
    return (
      <div className="px-3 py-2 text-sm font-medium text-slate-400 rounded-lg flex items-center gap-2 border border-slate-200 bg-white">
        <UserCheck className="h-4 w-4" />
        <span className="hidden sm:inline">Loading…</span>
      </div>
    );
  }

  // Real session — show static identity chip, no switcher.
  if (isRealSession) {
    return <UserChip role={role} name={user.full_name} />;
  }

  // Dev mode — keep the switcher dropdown.
  const Icon = ROLE_ICON[role];
  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "px-3 py-2 text-sm font-medium rounded-lg flex items-center gap-2",
          "border border-slate-200 bg-white hover:bg-slate-50 hover:border-slate-300",
          "transition-all duration-150"
        )}
        title={`Dev mode — signed in as ${user.full_name}`}
      >
        <Icon className="h-4 w-4 text-blue-600" />
        <span className="hidden sm:inline text-slate-700">{ROLE_LABELS[role]}</span>
        <span className="hidden lg:inline text-slate-400 font-normal">· {user.full_name.split(' ').slice(-1)[0]}</span>
        <ChevronDown className={cn("h-3.5 w-3.5 text-slate-400 transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div
          className="absolute right-0 mt-2 w-64 bg-white border border-slate-200 rounded-xl shadow-lg overflow-hidden animate-fadeInDown z-40"
          role="menu"
        >
          <div className="px-3 py-2 border-b border-slate-100 bg-amber-50/60">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-amber-600">Dev mode — switch role</p>
          </div>
          <ul className="py-1">
            {ROLES.map((r) => {
              const u = DEMO_USERS[r];
              const RIcon = ROLE_ICON[r];
              const active = r === role;
              return (
                <li key={r}>
                  <button
                    onClick={() => { setRole(r); setOpen(false); }}
                    className={cn(
                      "w-full flex items-center gap-3 px-3 py-2 text-left text-sm transition-colors",
                      active ? "bg-blue-50" : "hover:bg-slate-50"
                    )}
                    role="menuitem"
                  >
                    <RIcon className={cn("h-4 w-4 shrink-0", active ? "text-blue-600" : "text-slate-400")} />
                    <div className="min-w-0 flex-1">
                      <p className={cn("font-medium truncate", active ? "text-blue-700" : "text-slate-800")}>
                        {ROLE_LABELS[r]}
                      </p>
                      <p className="text-[11px] text-slate-400 truncate">{u.full_name}</p>
                    </div>
                    {active && <span className="text-[10px] font-bold text-blue-600">●</span>}
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
