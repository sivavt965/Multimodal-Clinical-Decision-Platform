'use client';

import React from 'react';
import { useToastStore } from '@/store/toastStore';
import { XCircle, CheckCircle, Info, X } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

const ICONS = {
  success: <CheckCircle className="w-5 h-5 text-emerald-500" />,
  error: <XCircle className="w-5 h-5 text-red-500" />,
  info: <Info className="w-5 h-5 text-blue-500" />
};

export function ToastContainer() {
  const toasts = useToastStore((state) => state.toasts);
  const removeToast = useToastStore((state) => state.removeToast);

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={cn(
            "flex items-start gap-3 p-4 bg-white border rounded-xl shadow-lg w-80 animate-in slide-in-from-right-8 fade-in duration-300",
            toast.type === 'error' && "border-red-200",
            toast.type === 'success' && "border-emerald-200",
            toast.type === 'info' && "border-blue-200"
          )}
        >
          <div className="shrink-0 mt-0.5">{ICONS[toast.type]}</div>
          <div className="flex-1">
            <h4 className="text-sm font-semibold text-gray-900">{toast.title}</h4>
            {toast.message && (
              <p className="text-sm text-gray-500 mt-1 leading-snug">{toast.message}</p>
            )}
          </div>
          <button
            onClick={() => removeToast(toast.id)}
            className="shrink-0 p-1 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      ))}
    </div>
  );
}
