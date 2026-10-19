import Markdown from 'react-markdown'
import type { ChatMessage } from '../useChat'

interface Props {
  message: ChatMessage
  debug: boolean
  onRate: (id: string, value: 1 | -1) => void
}

function Thumb({ down = false }: { down?: boolean }) {
  return (
    <svg viewBox="0 0 20 20" fill="currentColor" className={`h-4 w-4 ${down ? 'rotate-180' : ''}`} aria-hidden>
      <path d="M2 10.5a1.5 1.5 0 1 1 3 0v6a1.5 1.5 0 0 1-3 0v-6ZM6 10.33v5.43a2 2 0 0 0 1.1 1.79l.05.02A4 4 0 0 0 8.94 18h5.42a2 2 0 0 0 1.96-1.61l1.2-6A2 2 0 0 0 15.56 8H12V4a2 2 0 0 0-2-2 1 1 0 0 0-1 1v.67a4 4 0 0 1-.8 2.4L6.8 7.93a4 4 0 0 0-.8 2.4Z" />
    </svg>
  )
}

export default function Message({ message: m, debug, onRate }: Props) {
  if (m.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-mh-navy px-4 py-2 text-white">
          {m.text}
        </div>
      </div>
    )
  }

  if (m.role === 'error') {
    return (
      <div className="max-w-[80%] rounded-2xl rounded-bl-sm border border-red-200 bg-red-50 px-4 py-2 text-red-800">
        {m.text}
      </div>
    )
  }

  const canRate = m.messageId !== undefined && m.meta?.outcome !== 'greeting'

  return (
    <div className="max-w-[80%] space-y-2">
      <div className="rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-2 shadow-sm">
        {/* react-markdown never renders raw HTML, so LLM output can't inject markup. */}
        <div className="prose prose-sm prose-slate max-w-none">
          <Markdown components={{ a: props => <a {...props} target="_blank" rel="noopener noreferrer" /> }}>
            {m.text}
          </Markdown>
        </div>

        {m.sources && m.sources.length > 0 && (
          <div className="mt-3 border-t border-slate-100 pt-2">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Sources</p>
            <ul className="mt-1 space-y-1 text-sm">
              {m.sources.map(s => (
                <li key={s.url}>
                  <a href={s.click_url} target="_blank" rel="noopener noreferrer"
                     className="text-sky-700 underline-offset-2 hover:underline">
                    {s.title}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {canRate && (
        <div className="flex items-center gap-2 pl-1 text-xs text-slate-500">
          {m.vote ? (
            <span>Thanks for the feedback.</span>
          ) : (
            <>
              <button onClick={() => onRate(m.id, 1)} aria-label="Helpful"
                      className="flex items-center gap-1 rounded-full border border-slate-200 px-2 py-1 hover:bg-slate-100">
                <Thumb /> Helpful
              </button>
              <button onClick={() => onRate(m.id, -1)} aria-label="Not helpful"
                      className="flex items-center gap-1 rounded-full border border-slate-200 px-2 py-1 hover:bg-slate-100">
                <Thumb down /> Not helpful
              </button>
            </>
          )}
        </div>
      )}

      {debug && m.meta && (
        <p className="pl-1 font-mono text-xs text-slate-400">
          route={m.meta.route} | decision={m.meta.decision || '-'} | outcome={m.meta.outcome} | {m.meta.latency_s}s
        </p>
      )}
    </div>
  )
}
