'use client';

import React, { useEffect } from 'react';
import { Activity, Users, Settings, BookOpen, LogOut } from 'lucide-react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { RoleSwitcher } from '@/components/shared/RoleSwitcher';
import { useUserRole } from '@/lib/userRole';
import { useAuth } from '@/lib/auth';
import type { UserRole } from '@/lib/types';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

const NAV_LINKS: Array<{
  href: string;
  label: string;
  icon: typeof Users;
  /** If set, link is hidden unless current role is in this list. */
  visibleTo?: UserRole[];
}> = [
  { href: '/dashboard', label: 'Active Cases', icon: Users },
  { href: '/about',     label: 'About',        icon: BookOpen },
  { href: '/admin',     label: 'Admin',        icon: Settings, visibleTo: ['system_admin'] },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname() || '';
  const router = useRouter();
  const { role, hydrated } = useUserRole();
  const { session, loading: authLoading, signOut } = useAuth();

  // Redirect to /login if auth has resolved and there's no session.
  useEffect(() => {
    if (!authLoading && !session) {
      router.replace('/login');
    }
  }, [authLoading, session, router]);

  // Pre-hydration we render with the default role to keep SSR markup stable;
  // /admin link is hidden until we know the role on the client.
  const visibleNav = NAV_LINKS.filter((link) =>
    !link.visibleTo || (hydrated && link.visibleTo.includes(role))
  );

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Top Navigation Bar */}
      <header className="bg-white/85 backdrop-blur-md border-b border-slate-200/80 sticky top-0 z-30 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            {/* Logo & Title */}
            <Link href="/dashboard" className="flex items-center gap-3 group">
              <div className="bg-gradient-to-br from-blue-600 to-blue-700 p-2 rounded-lg shadow-sm group-hover:shadow-md group-hover:scale-105 transition-all duration-200">
                <Activity className="h-5 w-5 text-white" />
              </div>
              <div className="flex flex-col leading-tight">
                <span className="text-base font-bold text-slate-900 tracking-tight">
                  Multimodal CDS
                </span>
                <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">
                  Symile-MIMIC Platform
                </span>
              </div>
            </Link>

            {/* Nav Links */}
            <nav className="hidden md:flex items-center gap-1">
              {visibleNav.map(({ href, label, icon: Icon }) => {
                const active = pathname === href || pathname.startsWith(href + '/');
                return (
                  <Link
                    key={href}
                    href={href}
                    className={cn(
                      "relative px-3 py-2 text-sm font-medium rounded-lg flex items-center gap-2 transition-all duration-150",
                      active
                        ? "text-blue-700 bg-blue-50"
                        : "text-slate-500 hover:text-slate-900 hover:bg-slate-100"
                    )}
                  >
                    <Icon className={cn("h-4 w-4 transition-transform", active && "scale-110")} />
                    {label}
                    {active && (
                      <span className="absolute -bottom-[17px] left-1/2 -translate-x-1/2 h-[2px] w-8 bg-blue-600 rounded-full" />
                    )}
                  </Link>
                );
              })}
              <div className="w-px h-5 bg-slate-200 mx-3" />
              <RoleSwitcher />
              <button
                onClick={signOut}
                className="ml-1 px-3 py-2 text-sm font-medium text-red-600 hover:text-white hover:bg-red-600 rounded-lg flex items-center gap-2 transition-all duration-150"
              >
                <LogOut className="h-4 w-4" />
                Logout
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 animate-fadeIn">
        {children}
      </main>
    </div>
  );
}
