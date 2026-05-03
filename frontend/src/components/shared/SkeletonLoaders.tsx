import React from 'react';

export function DashboardSkeleton() {
  return (
    <div className="flex flex-col gap-2.5 animate-pulse">
      {[1, 2, 3, 4, 5, 6, 7].map((i) => (
        <div key={i} className="bg-white border border-gray-100 border-l-4 border-l-gray-200 rounded-xl shadow-[0_1px_3px_rgba(0,0,0,0.06)] px-5 py-4">
          <div className="flex items-center gap-4">
            {/* Avatar */}
            <div className="w-11 h-11 rounded-xl bg-gray-200 shrink-0" />
            {/* Name + meta */}
            <div className="flex-1 min-w-0 space-y-2">
              <div className="h-4 bg-gray-200 rounded w-40" />
              <div className="h-3 bg-gray-100 rounded w-56" />
            </div>
            {/* Risk badge */}
            <div className="hidden sm:block h-6 w-24 bg-gray-100 rounded-full" />
            {/* Finding bar */}
            <div className="hidden md:flex flex-col gap-1.5 w-44">
              <div className="h-3 bg-gray-100 rounded w-32" />
              <div className="h-1.5 bg-gray-100 rounded-full w-full" />
            </div>
            {/* Buttons */}
            <div className="flex items-center gap-2 ml-2">
              <div className="h-8 w-16 bg-gray-200 rounded-lg" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export function WorkspaceSkeleton() {
  return (
    <div className="flex h-[calc(100vh-130px)] gap-6 p-6 animate-pulse">
      {/* Left Column */}
      <div className="w-80 flex flex-col gap-6 shrink-0">
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm h-64 p-5" />
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm flex-1 p-5" />
      </div>
      
      {/* Middle Column */}
      <div className="flex-1 bg-white border border-gray-200 rounded-xl shadow-sm p-2" />
      
      {/* Right Column */}
      <div className="w-80 flex flex-col gap-6 shrink-0">
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm flex-1 p-5" />
      </div>
    </div>
  );
}
