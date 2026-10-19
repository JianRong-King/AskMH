import { useCallback, useEffect, useRef, useState } from 'react'
import { API_URL, ApiError, endSession, getGreeting, sendChat, sendFeedback, type ChatResponse, type Source } from './api'
import { clearToken } from './token'

export interface ChatMessage {
  id: string
  role: 'user' | 'bot' | 'error'
  text: string
  sources?: Source[]
  // Set only on the last reply of a turn: it is what /feedback rates.
  messageId?: number
  meta?: Pick<ChatResponse, 'route' | 'decision' | 'outcome' | 'latency_s'>
  vote?: 1 | -1
}

const BUSY = 'Sorry, the assistant is temporarily busy. Please try again in a minute.'
const UNREACHABLE = `Can't reach the AskMH server at ${API_URL || 'this site'}. Is the API running?`

const uid = () => crypto.randomUUID()

function errorMessage(e: unknown): string {
  if (e instanceof ApiError) {
    // 429 (quota) and 401 (session) already carry a message written for the user.
    if (e.status === 429 || e.status === 401) return e.message
    if (e.status === 503) return BUSY
  }
  return UNREACHABLE
}

async function fetchGreeting(): Promise<ChatMessage[]> {
  try {
    const g = await getGreeting()
    return [{ id: uid(), role: 'bot', text: `${g.greeting}\n\n${g.scope}` }]
  } catch (e) {
    return [{ id: uid(), role: 'error', text: errorMessage(e) }]
  }
}

export function useChat() {
  // Nothing is persisted: a page refresh starts a brand-new conversation.
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [messagesLeft, setMessagesLeft] = useState<number | null>(null)
  const session = useRef<string | null>(null)

  useEffect(() => {
    session.current = sessionId
  }, [sessionId])

  // Delete the server-side conversation on refresh, tab close or leaving the chat.
  useEffect(() => {
    const end = () => {
      if (session.current) endSession(session.current)
      session.current = null
    }
    window.addEventListener('pagehide', end)
    return () => {
      window.removeEventListener('pagehide', end)
      end()
    }
  }, [])

  // Only fill an empty chat: StrictMode runs effects twice in dev, and the user may have typed already.
  const greet = useCallback(() => {
    fetchGreeting().then(first => setMessages(prev => (prev.length ? prev : first)))
  }, [])

  useEffect(() => {
    greet()
  }, [greet])

  const send = useCallback(async (text: string) => {
    const message = text.trim()
    if (!message || loading) return
    setMessages(prev => [...prev, { id: uid(), role: 'user', text: message }])
    setLoading(true)
    try {
      const res = await sendChat(message, sessionId)
      setSessionId(res.session_id)
      setMessagesLeft(res.messages_left)
      const meta = { route: res.route, decision: res.decision, outcome: res.outcome, latency_s: res.latency_s }
      const replies: ChatMessage[] = res.replies.map((reply, i) =>
        i === res.replies.length - 1
          ? { id: uid(), role: 'bot', text: reply, sources: res.sources, messageId: res.message_id, meta }
          : { id: uid(), role: 'bot', text: reply },
      )
      setMessages(prev => [...prev, ...replies])
    } catch (e) {
      setMessages(prev => [...prev, { id: uid(), role: 'error', text: errorMessage(e) }])
      // An expired session can only be fixed by signing in again, which lives above this hook.
      if (e instanceof ApiError && e.status === 401) {
        clearToken()
        setTimeout(() => window.location.reload(), 1500)
      }
    } finally {
      setLoading(false)
    }
  }, [loading, sessionId])

  const rate = useCallback(async (id: string, value: 1 | -1) => {
    const target = messages.find(m => m.id === id)
    if (!sessionId || target?.messageId === undefined || target.vote) return
    const setVote = (vote: 1 | -1 | undefined) =>
      setMessages(prev => prev.map(m => (m.id === id ? { ...m, vote } : m)))
    setVote(value)
    try {
      await sendFeedback(sessionId, target.messageId, value)
    } catch {
      setVote(undefined)
    }
  }, [messages, sessionId])

  const newChat = useCallback(() => {
    if (sessionId) endSession(sessionId)
    setSessionId(null)
    setMessages([])
    greet()
  }, [greet, sessionId])

  return { messages, loading, send, rate, newChat, messagesLeft }
}
