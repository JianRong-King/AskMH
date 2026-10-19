// Typed client for the FastAPI backend. Types mirror api/schemas.py.

import { getToken } from './token'

// In production the API serves this bundle, so requests are same-origin. In dev, Vite
// serves the SPA on 5173 and the API lives on 8001.
const DEFAULT_API_URL = import.meta.env.DEV ? 'http://127.0.0.1:8001' : ''

export const API_URL = (import.meta.env.VITE_API_URL ?? DEFAULT_API_URL).replace(/\/$/, '')

export type Outcome = 'answered' | 'clarify' | 'declined' | 'handoff' | 'greeting'

export interface Source {
  title: string
  url: string
  click_url: string
}

export interface ChatResponse {
  session_id: string
  message_id: number
  replies: string[]
  sources: Source[]
  route: string
  decision: string
  outcome: Outcome
  latency_s: number
  messages_left: number | null
}

export interface Greeting {
  greeting: string
  scope: string
}

export interface AppConfig {
  auth_enabled: boolean
  google_client_id: string
}

export interface LoginResult {
  token: string
  expires_in: number
  email: string
  name: string
}

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = {}
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`

  const init: RequestInit = { headers }
  if (body !== undefined) {
    init.method = 'POST'
    headers['Content-Type'] = 'application/json'
    init.body = JSON.stringify(body)
  }

  const res = await fetch(`${API_URL}/api${path}`, init)
  if (!res.ok) {
    const data = await res.json().catch(() => null)
    throw new ApiError(res.status, typeof data?.detail === 'string' ? data.detail : res.statusText)
  }
  return (res.status === 204 ? undefined : await res.json()) as T
}

export const getConfig = () => request<AppConfig>('/config')

export const loginWithGoogle = (credential: string) =>
  request<LoginResult>('/auth/google', { credential })

export const getGreeting = () => request<Greeting>('/greeting')

export const sendChat = (message: string, sessionId: string | null) =>
  request<ChatResponse>('/chat', { message, session_id: sessionId })

export const sendFeedback = (sessionId: string, messageId: number, value: 1 | -1) =>
  request<void>('/feedback', { session_id: sessionId, message_id: messageId, value })

export interface StatsRow {
  ts: string
  question: string
  route?: string
  outcome: Outcome | 'error'
  latency_s?: number
}

export interface Stats {
  days: number
  messages: number
  sessions: number
  avg_latency_s: number | null
  outcomes: Record<string, number>
  resolution_rate: number | null
  csat: number | null
  feedback_count: number
  source_ctr: number | null
  per_day: { day: string; messages: number }[]
  recent: StatsRow[]
  unanswered: StatsRow[]
}

export const getStats = (days: number) => request<Stats>(`/admin/stats?days=${days}`)

// keepalive lets the request finish while the page is being unloaded.
export function endSession(sessionId: string) {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`
  fetch(`${API_URL}/api/session/end`, {
    method: 'POST', headers, keepalive: true, body: JSON.stringify({ session_id: sessionId }),
  }).catch(() => {})
}
