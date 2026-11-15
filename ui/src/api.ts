import type { ClaimResult } from './types'

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    const detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  submit: (message: string, attachment_text: string | null) =>
    call<ClaimResult>('/claims', { method: 'POST', body: JSON.stringify({ message, attachment_text }) }),
  review: (runId: string, approved: boolean, reviewer: string, note: string) =>
    call<ClaimResult>(`/claims/${runId}/review`, {
      method: 'POST', body: JSON.stringify({ approved, reviewer, note: note || null }),
    }),
  health: () => call<{ status: string; provider: string }>('/health'),
}
