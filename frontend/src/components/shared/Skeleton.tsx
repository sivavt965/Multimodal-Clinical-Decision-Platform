'use client';

import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

/**
 * Animated skeleton placeholder — replaces full-page spinners
 * with per-component loading indicators.
 */
export function Skeleton({
  className,
  variant = 'rect',
}: {
  className?: string;
  variant?: 'rect' | 'circle' | 'text';
}) {
  return (
    <div
      className={cn(
        "animate-pulse bg-gradient-to-r from-gray-200 via-gray-100 to-gray-200 bg-[length:200%_100%]",
        variant === 'circle' && "rounded-full",
        variant === 'text' && "rounded h-4",
        variant === 'rect' && "rounded-lg",
        className
      )}
    />
  );
}

/** Skeleton for a CXR image block */
export function CXRSkeleton() {
  return (
    <div className="h-full bg-[#0f172a] rounded-xl border border-gray-800 p-4 flex flex-col items-center justify-center gap-4">
      <Skeleton className="w-64 h-64 bg-gray-700/50" />
      <div className="space-y-2 w-48">
        <Skeleton className="h-3 bg-gray-700/50 w-full" variant="text" />
        <Skeleton className="h-3 bg-gray-700/50 w-3/4" variant="text" />
      </div>
      <p className="text-xs text-slate-500 mt-2 animate-pulse">
        Running DenseNet121 inference…
      </p>
    </div>
  );
}

/** Skeleton for a prediction summary panel */
export function PredictionSkeleton() {
  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-5 space-y-4">
      <Skeleton className="h-5 w-40" variant="text" />
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="flex items-center justify-between">
          <Skeleton className="h-4 w-24" variant="text" />
          <Skeleton className="h-4 w-16" variant="text" />
        </div>
      ))}
    </div>
  );
}

/** Skeleton for an ECG parameter card */
export function ECGSkeleton() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="flex items-center justify-between">
          <Skeleton className="h-4 w-32" variant="text" />
          <Skeleton className="h-8 w-24 rounded-lg" />
        </div>
      ))}
    </div>
  );
}

/** Generic card skeleton with configurable rows */
export function CardSkeleton({ rows = 4 }: { rows?: number }) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-5 space-y-3">
      <Skeleton className="h-5 w-32 mb-4" variant="text" />
      {Array.from({ length: rows }).map((_, i) => (
        <Skeleton key={i} className="h-4 w-full" variant="text" />
      ))}
    </div>
  );
}
