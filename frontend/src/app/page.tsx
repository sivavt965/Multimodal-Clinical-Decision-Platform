import { redirect } from 'next/navigation'

/**
 * Root route — immediately redirects to the clinical dashboard.
 * This is a server component (no 'use client'), so the redirect
 * happens before any HTML is sent to the browser.
 */
export default function RootPage() {
  redirect('/dashboard')
}
