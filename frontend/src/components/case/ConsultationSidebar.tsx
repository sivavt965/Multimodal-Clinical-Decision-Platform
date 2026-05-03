'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useCaseStore } from '@/store/caseStore';
import { fetchCaseDetail } from '@/lib/api';
import { X, Send, User, Stethoscope, Clock, CheckCheck, Check, MessageSquarePlus } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

const POLL_INTERVAL_MS = 5000;

const ROLE_CONFIG = {
  ward_doctor: {
    label: 'Ward Doctor',
    short: 'WD',
    bubble: 'bg-white text-gray-800 border border-gray-200 rounded-tl-sm',
    header: 'text-emerald-700',
    avatar: 'bg-emerald-500',
    align: 'items-start',
    headerRow: 'flex-row',
    sendBtn: 'bg-emerald-500 hover:bg-emerald-600',
    placeholder: 'Write clinical note as Ward Doctor…',
  },
  radiologist: {
    label: 'Radiologist',
    short: 'R',
    bubble: 'bg-blue-600 text-white rounded-tr-sm',
    header: 'text-blue-700',
    avatar: 'bg-blue-600',
    align: 'items-end',
    headerRow: 'flex-row-reverse',
    sendBtn: 'bg-blue-600 hover:bg-blue-700',
    placeholder: 'Write radiologist finding or clinical note…',
  },
} as const;

export function ConsultationSidebar() {
  const isSidebarOpen = useCaseStore((state) => state.isSidebarOpen);
  const closeSidebar  = useCaseStore((state) => state.closeSidebar);
  const currentCase   = useCaseStore((state) => state.currentCase);
  const sendMessage   = useCaseStore((state) => state.sendMessage);

  const [input, setInput] = useState('');
  const [senderRole, setSenderRole] = useState<'ward_doctor' | 'radiologist'>('radiologist');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const messages = currentCase?.consultation?.messages || [];
  const unreadCount = messages.filter(m => !m.read).length;

  // Poll for new messages while open
  useEffect(() => {
    if (!isSidebarOpen || !currentCase) return;
    const caseId = currentCase.case.id;
    let active = true;
    const poll = async () => {
      try {
        const detail = await fetchCaseDetail(caseId);
        if (!active) return;
        const cur = useCaseStore.getState().currentCase?.consultation?.messages?.length ?? 0;
        if ((detail.consultation?.messages?.length ?? 0) > cur) {
          useCaseStore.setState({ currentCase: detail });
        }
      } catch { /* silently retry */ }
    };
    const timer = setInterval(poll, POLL_INTERVAL_MS);
    return () => { active = false; clearInterval(timer); };
  }, [isSidebarOpen, currentCase?.case?.id]);

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length]);

  // Focus textarea when sidebar opens
  useEffect(() => {
    if (isSidebarOpen) {
      setTimeout(() => textareaRef.current?.focus(), 300);
    }
  }, [isSidebarOpen]);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !currentCase) return;
    const trimmed = input.trim();
    setInput('');
    await sendMessage({
      id: Math.random().toString(36).substring(2, 9),
      role: senderRole,
      type: 'text',
      content: trimmed,
      sent_at: new Date().toISOString(),
      read: false,
    });
  };

  return (
    <div
      className={cn(
        // fixed: takes sidebar out of any stacking context / overflow-hidden parent
        // top-16 = 64px nav height so it starts below the nav bar
        "fixed right-0 top-16 bottom-0 w-96 bg-white border-l border-gray-200 shadow-xl flex flex-col z-50 transform transition-transform duration-300 ease-in-out",
        isSidebarOpen ? "translate-x-0" : "translate-x-full"
      )}
    >
      {/* ── Header ── */}
      <div className="px-4 py-3 border-b border-gray-100 flex justify-between items-center bg-gradient-to-r from-slate-50 to-blue-50 shrink-0">
        <div className="flex items-center gap-2.5">
          <div className="bg-blue-600 text-white p-1.5 rounded-lg">
            <MessageSquarePlus className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-bold text-gray-900 text-sm">Consultation Thread</h3>
            <p className="text-xs text-gray-500">
              {currentCase?.patient.first_name} — {messages.length} message{messages.length !== 1 ? 's' : ''}
              {unreadCount > 0 && (
                <span className="ml-1 bg-red-500 text-white text-[9px] font-bold px-1.5 py-0.5 rounded-full">
                  {unreadCount} new
                </span>
              )}
            </p>
          </div>
        </div>
        <button
          onClick={(e) => { e.stopPropagation(); closeSidebar(); }}
          className="p-1.5 text-gray-400 hover:text-gray-700 hover:bg-gray-200 rounded-lg transition-colors"
          aria-label="Close consultation sidebar"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* ── Message List ── */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50/80">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-12">
            <div className="bg-white border-2 border-dashed border-gray-200 rounded-full p-5 mb-3">
              <MessageSquarePlus className="w-7 h-7 text-gray-300" />
            </div>
            <p className="text-sm font-semibold text-gray-500">No messages yet</p>
            <p className="text-xs text-gray-400 mt-1">Start the consultation below</p>
          </div>
        ) : (
          messages.map((msg, idx) => {
            const cfg = ROLE_CONFIG[msg.role as keyof typeof ROLE_CONFIG] ?? ROLE_CONFIG.ward_doctor;
            const time = new Date(msg.sent_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const dateStr = new Date(msg.sent_at).toLocaleDateString([], { month: 'short', day: 'numeric' });

            const prev = idx > 0 ? messages[idx - 1] : null;
            const showDateSep = !prev || new Date(prev.sent_at).toDateString() !== new Date(msg.sent_at).toDateString();

            return (
              <React.Fragment key={msg.id}>
                {showDateSep && (
                  <div className="flex items-center gap-2 my-3">
                    <div className="flex-1 h-px bg-gray-200" />
                    <span className="text-[10px] text-gray-400 font-medium bg-gray-50 px-2">{dateStr}</span>
                    <div className="flex-1 h-px bg-gray-200" />
                  </div>
                )}
                <div className={cn("flex flex-col animate-fadeInUp", cfg.align)}>
                  <div className={cn("flex items-center gap-1.5 mb-1", cfg.headerRow)}>
                    <div className={cn("w-6 h-6 rounded-full flex items-center justify-center text-white text-[9px] font-bold shrink-0 shadow-sm", cfg.avatar)}>
                      {cfg.short}
                    </div>
                    <span className={cn("text-[10px] font-bold truncate max-w-[120px]", cfg.header)}>
                      {cfg.label}
                    </span>
                    <span className="text-[10px] text-gray-400 flex items-center gap-0.5 shrink-0">
                      <Clock className="w-2.5 h-2.5" />{time}
                    </span>
                  </div>

                  <div className={cn("px-3.5 py-2 rounded-2xl text-sm max-w-[85%] leading-relaxed shadow-sm", cfg.bubble)}>
                    {typeof msg.content === 'string' ? msg.content : '[System message]'}
                  </div>

                  <div className={cn("flex items-center gap-0.5 mt-0.5", cfg.headerRow)}>
                    {msg.read
                      ? <CheckCheck className="w-3 h-3 text-blue-400" />
                      : <Check className="w-3 h-3 text-gray-300" />}
                    <span className="text-[9px] text-gray-300">{msg.read ? 'Read' : 'Sent'}</span>
                  </div>
                </div>
              </React.Fragment>
            );
          })
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* ── Role Selector + Input ── */}
      <div className="border-t border-gray-200 bg-white shrink-0">
        {/* Sender role toggle */}
        <div className="px-4 pt-3 pb-2 flex items-center gap-2">
          <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider shrink-0">Sending as:</span>
          <div className="flex bg-gray-100 rounded-lg p-0.5 gap-0.5 w-full">
            <button
              type="button"
              onClick={() => setSenderRole('ward_doctor')}
              className={cn(
                "flex-1 flex items-center justify-center gap-1 py-1.5 rounded-md text-xs font-semibold transition-all",
                senderRole === 'ward_doctor'
                  ? "bg-emerald-500 text-white shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              )}
            >
              <User className="w-3 h-3" />
              Ward Doctor
            </button>
            <button
              type="button"
              onClick={() => setSenderRole('radiologist')}
              className={cn(
                "flex-1 flex items-center justify-center gap-1 py-1.5 rounded-md text-xs font-semibold transition-all",
                senderRole === 'radiologist'
                  ? "bg-blue-600 text-white shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              )}
            >
              <Stethoscope className="w-3 h-3" />
              Radiologist
            </button>
          </div>
        </div>

        {/* Message input */}
        <div className="px-4 pb-4">
          <form onSubmit={handleSend} className="relative">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={ROLE_CONFIG[senderRole].placeholder}
              rows={3}
              className="w-full border border-gray-200 rounded-xl pl-3 pr-10 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 resize-none transition-colors"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend(e as any);
                }
              }}
            />
            <button
              type="submit"
              disabled={!input.trim()}
              className={cn(
                "absolute bottom-3 right-3 p-1.5 text-white rounded-lg transition-colors disabled:bg-gray-200 disabled:cursor-not-allowed",
                ROLE_CONFIG[senderRole].sendBtn
              )}
              title="Send (Enter)"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          <p className="text-[10px] text-gray-400 mt-1 text-right">Enter to send · Shift+Enter for newline</p>
        </div>
      </div>
    </div>
  );
}
