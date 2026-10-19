import { useState } from 'react'
import { TOPICS, type Topic, type TopicIcon } from '../topics'

interface Props {
  onAsk: (question: string) => void
}

const ICONS: Record<TopicIcon, string> = {
  bag: 'M9 7V5a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2M6 7h12a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1Zm3 0v13m6-13v13',
  card: 'M3 7a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Zm0 3h18M7 15h3',
  ticket: 'M4 7a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v2a2 2 0 0 0 0 4v2a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-2a2 2 0 0 0 0-4V7Zm10-2v12',
  refund: 'M4 12a8 8 0 1 0 2.34-5.66M4 4v4h4M12 8v4l3 2',
  star: 'm12 3 2.6 5.6 6.1.7-4.5 4.2 1.2 6L12 16.6 6.6 19.5l1.2-6-4.5-4.2 6.1-.7L12 3Z',
  assist: 'M12 5a2 2 0 1 0 0-4 2 2 0 0 0 0 4Zm-1 3h3l1 6h-4m-1-6-1 5a5 5 0 1 0 7 5',
  passport: 'M6 3h11a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Zm6 4a3 3 0 1 0 0 6 3 3 0 0 0 0-6Zm-3 10h6',
}

function Icon({ name, className = 'h-6 w-6' }: { name: TopicIcon; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.6}
         strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden>
      <path d={ICONS[name]} />
    </svg>
  )
}

export default function Suggestions({ onAsk }: Props) {
  const [open, setOpen] = useState<Topic | null>(null)

  return (
    <div className="animate-rise space-y-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Popular topics</p>
      <div className="flex flex-wrap gap-2">
        {TOPICS.map(t => {
          const active = open?.title === t.title
          return (
            <button key={t.title} onClick={() => setOpen(active ? null : t)} aria-pressed={active}
                    className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-sm transition ${
                      active
                        ? 'border-mh-navy bg-mh-navy text-white'
                        : 'border-slate-200 bg-white text-slate-700 hover:border-mh-navy/40 hover:bg-mh-navy/5'
                    }`}>
              <Icon name={t.icon} className="h-4 w-4" />
              {t.title}
            </button>
          )
        })}
      </div>
      {open && (
        <ul key={open.title} className="animate-rise space-y-2">
          {open.questions.map(q => (
            <li key={q}>
              <button onClick={() => onAsk(q)}
                      className="group flex w-full items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-2.5 text-left text-sm shadow-sm transition hover:border-mh-navy/40 hover:shadow">
                <span>{q}</span>
                <span className="text-mh-red transition-transform group-hover:translate-x-1" aria-hidden>&rarr;</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
