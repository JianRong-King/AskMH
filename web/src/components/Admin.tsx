import { useEffect, useState } from 'react'
import { ApiError, getStats, type Stats, type StatsRow } from '../api'

const RANGES = [7, 14, 30, 90]
const OUTCOME_COLORS: Record<string, string> = {
  answered: 'bg-emerald-500',
  clarify: 'bg-sky-500',
  declined: 'bg-amber-500',
  handoff: 'bg-mh-red',
  greeting: 'bg-slate-300',
  error: 'bg-slate-500',
}

const pct = (v: number | null) => (v === null ? '-' : `${Math.round(v * 100)}%`)
const when = (ts: string) =>
  new Date(ts).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })

function Tile({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-mh-navy">{value}</p>
      {hint && <p className="mt-0.5 text-xs text-slate-400">{hint}</p>}
    </div>
  )
}

function Questions({ title, rows, empty }: { title: string; rows: StatsRow[]; empty: string }) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <h2 className="border-b border-slate-100 px-4 py-3 font-medium">{title}</h2>
      {rows.length === 0 ? (
        <p className="px-4 py-6 text-sm text-slate-400">{empty}</p>
      ) : (
        <ul className="max-h-96 divide-y divide-slate-100 overflow-y-auto text-sm">
          {rows.map((r, i) => (
            <li key={`${r.ts}-${i}`} className="flex items-start gap-3 px-4 py-2.5">
              <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${OUTCOME_COLORS[r.outcome] ?? 'bg-slate-300'}`}
                    title={r.outcome} />
              <span className="min-w-0 flex-1 break-words">{r.question}</span>
              <span className="shrink-0 text-xs text-slate-400">{when(r.ts)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

export default function Admin({ onExit }: { onExit: () => void }) {
  const [days, setDays] = useState(14)
  const [stats, setStats] = useState<Stats | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let live = true
    getStats(days)
      .then(s => { if (live) { setStats(s); setError('') } })
      .catch(e => { if (live) setError(e instanceof ApiError ? e.message : 'Cannot reach the server') })
    return () => { live = false }
  }, [days])

  const peak = Math.max(1, ...(stats?.per_day.map(d => d.messages) ?? []))
  const outcomeTotal = Object.values(stats?.outcomes ?? {}).reduce((a, b) => a + b, 0)

  return (
    <div className="min-h-dvh bg-slate-50 text-slate-900">
      <header className="bg-mh-navy text-white">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-4 py-3">
          <div>
            <h1 className="text-lg font-semibold">AskMH admin</h1>
            <p className="text-xs text-white/70">Traffic and questions</p>
          </div>
          <div className="flex items-center gap-2 text-sm">
            {RANGES.map(r => (
              <button key={r} onClick={() => setDays(r)}
                      className={`rounded-full px-3 py-1 ${r === days ? 'bg-white text-mh-navy' : 'border border-white/30 hover:bg-white/10'}`}>
                {r}d
              </button>
            ))}
            <button onClick={onExit} className="ml-2 rounded-full border border-white/30 px-3 py-1 hover:bg-white/10">
              Back to chat
            </button>
          </div>
        </div>
      </header>

      <main className="animate-rise mx-auto max-w-5xl space-y-6 px-4 py-6">
        {error && <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">{error}</p>}
        {!stats && !error && <p className="text-sm text-slate-500">Loading...</p>}

        {stats && (
          <>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
              <Tile label="Messages" value={stats.messages} />
              <Tile label="Sessions" value={stats.sessions} />
              <Tile label="Resolution" value={pct(stats.resolution_rate)} hint="answered / attempts" />
              <Tile label="CSAT" value={pct(stats.csat)} hint={`${stats.feedback_count} votes`} />
              <Tile label="Source CTR" value={pct(stats.source_ctr)} hint="of answered" />
              <Tile label="Avg latency" value={stats.avg_latency_s === null ? '-' : `${stats.avg_latency_s}s`} />
            </div>

            <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <h2 className="font-medium">Messages per day</h2>
              {stats.per_day.length === 0 ? (
                <p className="py-6 text-sm text-slate-400">No traffic in this range.</p>
              ) : (
                <div className="mt-4 flex h-40 items-end gap-1 overflow-x-auto">
                  {stats.per_day.map(d => (
                    <div key={d.day} className="flex min-w-6 flex-1 flex-col items-center gap-1"
                         title={`${d.day}: ${d.messages}`}>
                      <span className="text-[10px] text-slate-500">{d.messages}</span>
                      <div className="w-full rounded-t bg-mh-navy" style={{ height: `${(d.messages / peak) * 110}px` }} />
                      <span className="text-[10px] text-slate-400">{d.day.slice(5)}</span>
                    </div>
                  ))}
                </div>
              )}

              {outcomeTotal > 0 && (
                <>
                  <h2 className="mt-6 font-medium">Outcomes</h2>
                  <div className="mt-3 flex h-3 overflow-hidden rounded-full bg-slate-100">
                    {Object.entries(stats.outcomes).map(([o, n]) => (
                      <div key={o} className={OUTCOME_COLORS[o] ?? 'bg-slate-300'}
                           style={{ width: `${(n / outcomeTotal) * 100}%` }} title={`${o}: ${n}`} />
                    ))}
                  </div>
                  <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
                    {Object.entries(stats.outcomes).map(([o, n]) => (
                      <span key={o} className="flex items-center gap-1.5">
                        <span className={`h-2 w-2 rounded-full ${OUTCOME_COLORS[o] ?? 'bg-slate-300'}`} />
                        {o} {n}
                      </span>
                    ))}
                  </div>
                </>
              )}
            </section>

            <div className="grid gap-6 lg:grid-cols-2">
              <Questions title="Recent questions" rows={stats.recent} empty="No questions yet." />
              <Questions title="Unanswered (knowledge gaps)" rows={stats.unanswered} empty="Nothing declined." />
            </div>
          </>
        )}
      </main>
    </div>
  )
}
