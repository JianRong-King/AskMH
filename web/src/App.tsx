import { useEffect, useRef, useState } from 'react'
import Admin from './components/Admin'
import Composer from './components/Composer'
import Message from './components/Message'
import Suggestions from './components/Suggestions'
import SignIn from './components/SignIn'
import TypingIndicator from './components/TypingIndicator'
import Welcome from './components/Welcome'
import { useAuth } from './useAuth'
import { useChat } from './useChat'

interface ChatProps {
  chat: ReturnType<typeof useChat>
  onBack: () => void
  onSignOut: () => void
  canSignOut: boolean
}

function Chat({ chat, onBack, onSignOut, canSignOut }: ChatProps) {
  const { messages, loading, send, rate, newChat, messagesLeft } = chat
  const [debug, setDebug] = useState(false)
  const bottom = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  return (
    <div className="flex h-dvh flex-col bg-slate-50 text-slate-900">
      <header className="bg-mh-navy text-white shadow-sm">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-4 px-4 py-3">
          <div className="flex items-center gap-3">
            <button onClick={onBack} aria-label="Back to home" title="Back to home"
                    className="-ml-2 rounded-full p-2 hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/40">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"
                   strokeLinejoin="round" className="h-5 w-5" aria-hidden>
                <path d="M15 18l-6-6 6-6" />
              </svg>
            </button>
            <div>
              <h1 className="text-lg font-semibold">AskMH</h1>
              <p className="text-xs text-white/70">Malaysia Airlines FAQ assistant</p>
            </div>
          </div>
          <div className="flex items-center gap-3 text-sm">
            {messagesLeft !== null && (
              <span className="text-xs text-white/70">{messagesLeft} left today</span>
            )}
            <label className="flex cursor-pointer items-center gap-1.5 text-white/80">
              <input type="checkbox" checked={debug} onChange={e => setDebug(e.target.checked)} />
              Debug
            </label>
            <button onClick={newChat} disabled={loading}
                    className="rounded-full border border-white/30 px-3 py-1 hover:bg-white/10 disabled:opacity-50">
              New chat
            </button>
            {canSignOut && (
              <button onClick={onSignOut}
                      className="rounded-full border border-white/30 px-3 py-1 hover:bg-white/10">
                Sign out
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="animate-rise flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-4 px-4 py-6">
          {messages.map(m => <Message key={m.id} message={m} debug={debug} onRate={rate} />)}
          {!loading && messages.length > 0 && !messages.some(m => m.role === 'user') && <Suggestions onAsk={send} />}
          {loading && <TypingIndicator />}
          <div ref={bottom} />
        </div>
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-3xl px-4 py-3">
          <Composer disabled={loading} onSend={send} />
          <p className="mt-2 text-center text-xs text-slate-400">
            Answers come only from the AskMH FAQ. Type "agent" to reach a human.
          </p>
        </div>
      </footer>
    </div>
  )
}

interface HomeProps {
  onSignOut: () => void
  canSignOut: boolean
}

// Owns the conversation so it survives going back to the welcome page.
function Home({ onSignOut, canSignOut }: HomeProps) {
  const chat = useChat()
  const [view, setView] = useState<'welcome' | 'chat'>('welcome')

  // Browser back/forward moves between welcome and chat. A refresh always starts on welcome.
  useEffect(() => {
    if (window.history.state?.view === 'chat') window.history.replaceState(null, '')
    const onPop = (e: PopStateEvent) => setView(e.state?.view === 'chat' ? 'chat' : 'welcome')
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const start = () => {
    window.history.pushState({ view: 'chat' }, '')
    setView('chat')
  }

  if (view === 'welcome') {
    return <Welcome onStart={start} resume={chat.messages.some(m => m.role === 'user')} />
  }

  return <Chat chat={chat} onBack={() => window.history.back()} onSignOut={onSignOut} canSignOut={canSignOut} />
}

export default function App() {
  const { state, config, error, buttonRef, signOut } = useAuth()
  const [admin, setAdmin] = useState(() => window.location.hash === '#admin')

  useEffect(() => {
    const onHash = () => setAdmin(window.location.hash === '#admin')
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  if (state === 'loading') {
    return (
      <div className="flex h-dvh items-center justify-center bg-slate-50 text-sm text-slate-500">
        Starting up...
      </div>
    )
  }

  if (state === 'unreachable') {
    return (
      <div className="flex h-dvh items-center justify-center bg-slate-50 px-4 text-center text-sm text-slate-600">
        Can't reach the AskMH server. If this is a hosted demo it may be waking up, so try
        again in a moment.
      </div>
    )
  }

  if (state === 'signed-out') return <SignIn buttonRef={buttonRef} error={error} />

  if (admin) return <Admin onExit={() => { window.location.hash = '' }} />

  return <Home onSignOut={signOut} canSignOut={Boolean(config?.auth_enabled)} />
}
