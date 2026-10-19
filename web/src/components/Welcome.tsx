interface Props {
  onStart: () => void
  resume: boolean
}

function Plane({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" className={className} aria-hidden>
      <path d="M21 15.5v-2l-8-5V3.5a1.5 1.5 0 0 0-3 0V8.5l-8 5v2l8-2.5V18l-2 1.5V21l3.5-1 3.5 1v-1.5L13 18v-5l8 2.5Z" />
    </svg>
  )
}

// Stepped Petronas tower outline centred on cx, ground at y=200. Each step is [half width, top].
const STEPS: [number, number][] = [[20, 80], [17, 62], [13, 48], [8, 38], [1, 6]]

function petronas(cx: number) {
  const left = STEPS.flatMap(([h, top], i) => [[cx - h, i ? STEPS[i - 1][1] : 200], [cx - h, top]])
  const right = left.map(([x, y]) => [2 * cx - x, y]).reverse()
  return `M${[...left, ...right].map(p => p.join(' ')).join(' L')} Z`
}

// The drawing is 15:1, wider than any screen strip it fills, so `slice` only ever crops the
// sides and the landmarks in the middle always keep their tops.
const SKY_W = 3000
const MID = SKY_W / 2

// Deterministic "random" low-rise blocks, kept clear of the landmarks.
const BUILDINGS: [number, number, number][] = (() => {
  const out: [number, number, number][] = []
  let seed = 7
  const rand = () => ((seed = (seed * 9301 + 49297) % 233280) / 233280)
  for (let x = 0; x < SKY_W; ) {
    const w = 35 + Math.round(rand() * 35)
    const nearTowers = x + w > MID - 110 && x < MID + 300
    if (!nearTowers) out.push([x, w, 110 + Math.round(rand() * 60)])
    x += w + 4 + Math.round(rand() * 10)
  }
  return out
})()

function Skyline({ className }: { className?: string }) {
  const kl = MID + 230
  return (
    <svg viewBox={`0 0 ${SKY_W} 200`} preserveAspectRatio="xMidYMax slice" className={className} aria-hidden>
      <g fill="currentColor">
        {BUILDINGS.map(([x, w, y]) => <rect key={x} x={x} y={y} width={w} height={200 - y} />)}
        <path d={petronas(MID - 30)} />
        <path d={petronas(MID + 30)} />
        <rect x={MID - 10} y="118" width="20" height="5" />
        <rect x={kl - 4} y="70" width="8" height="130" />
        <ellipse cx={kl} cy="62" rx="17" ry="9" />
        <rect x={kl - 4} y="40" width="8" height="16" />
        <rect x={kl - 1} y="8" width="2" height="34" />
      </g>
    </svg>
  )
}

function Hibiscus({ className }: { className?: string }) {
  const petal = 'M0 0 C -45 -25, -62 -95, -26 -122 C -14 -130, -4 -118, 0 -126 C 4 -118, 14 -130, 26 -122 C 62 -95, 45 -25, 0 0 Z'
  return (
    <svg viewBox="-140 -140 280 280" className={className} aria-hidden>
      {[0, 72, 144, 216, 288].map(a => (
        <path key={a} d={petal} transform={`rotate(${a})`}
              className="fill-mh-red/25 stroke-mh-red/70" strokeWidth="2" strokeLinejoin="round" />
      ))}
      <circle r="16" className="fill-mh-red/50" />
      <g transform="rotate(30)">
        <path d="M0 0 L0 -96" className="stroke-my-yellow/80" strokeWidth="3" strokeLinecap="round" />
        {[[-8, -98], [0, -106], [8, -98], [-6, -88], [6, -88]].map(([x, y]) => (
          <circle key={`${x},${y}`} cx={x} cy={y} r="4" className="fill-my-yellow" />
        ))}
      </g>
    </svg>
  )
}

const POINTS = [
  { title: 'Official FAQ only', text: 'Answers come straight from the AskMH help centre, nothing made up.' },
  { title: 'Cited sources', text: 'Every answer links to the article it came from.' },
  {
    title: 'Human backup',
    text: 'Live chat handoff, 7:00 am - 11:00 pm MYT.',
    badge: 'Simulated',
    note: 'Demo only: no human connected.',
  },
]

function Landing({ onNext, resume }: { onNext: () => void; resume: boolean }) {
  return (
    <div className="relative flex min-h-dvh flex-col overflow-hidden bg-gradient-to-br from-mh-navy via-mh-navy-light to-mh-navy text-white">
      <svg viewBox="0 0 800 400" preserveAspectRatio="none" aria-hidden
           className="pointer-events-none absolute inset-0 h-full w-full opacity-20">
        <path d="M-20 340 C 200 300, 420 120, 820 60" fill="none" stroke="white" strokeWidth="1.5" strokeDasharray="6 10" />
        <path d="M-20 380 C 260 360, 520 220, 820 180" fill="none" stroke="white" strokeWidth="1" strokeDasharray="2 8" />
      </svg>
      <div className="absolute -right-24 -top-24 h-72 w-72 rounded-full bg-mh-red/20 blur-3xl" aria-hidden />
      <Hibiscus className="animate-drift pointer-events-none absolute -right-20 -top-20 h-56 w-56 opacity-60 sm:-right-16 sm:-top-16 sm:h-96 sm:w-96 sm:opacity-80" />
      <Skyline className="pointer-events-none fixed inset-x-0 bottom-0 h-40 w-full text-white/[0.07] sm:h-56" />

      <main className="animate-rise relative mx-auto flex w-full max-w-4xl flex-1 flex-col justify-center px-6 pb-36 pt-16">
        <div className="flex items-center gap-2 text-sm font-medium uppercase tracking-[0.2em] text-my-yellow">
          <Plane className="h-4 w-4 rotate-45 text-white" />
          Travel help, anytime
        </div>
        <h1 className="mt-4 text-5xl font-semibold tracking-tight sm:text-6xl">AskMH</h1>
        <p className="mt-3 max-w-xl text-lg text-white/80">
          Your Malaysia Airlines FAQ assistant. Quick answers on baggage, check-in, Enrich, refunds and more.
        </p>

        <button onClick={onNext}
                className="group mt-10 inline-flex w-fit items-center gap-2 rounded-full bg-white px-7 py-3 font-medium text-mh-navy shadow-lg shadow-black/20 transition hover:bg-white/90 focus:outline-none focus-visible:ring-4 focus-visible:ring-white/40">
          {resume ? 'Continue chat' : 'Get started'}
          <span className="transition-transform group-hover:translate-x-1" aria-hidden>&rarr;</span>
        </button>

        <ul className="mt-14 grid gap-4 sm:grid-cols-3">
          {POINTS.map(p => (
            <li key={p.title} className="rounded-2xl border border-white/10 border-t-2 border-t-my-yellow/70 bg-white/5 p-4 backdrop-blur">
              <p className="flex items-center gap-2 font-medium text-my-yellow">
                {p.title}
                {p.badge && (
                  <span className="rounded-full border border-white/30 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-white/80">
                    {p.badge}
                  </span>
                )}
              </p>
              <p className="mt-1 text-sm text-white/70">{p.text}</p>
              {p.note && <p className="mt-1 text-xs italic text-white/50">{p.note}</p>}
            </li>
          ))}
        </ul>
      </main>

      <footer className="relative px-6 pb-6 text-center text-xs text-white/50">
        Portfolio project, not affiliated with Malaysia Airlines. FAQ content belongs to Malaysia Airlines.
        <br />
        Questions are logged anonymously to improve answers.
      </footer>
    </div>
  )
}

export default function Welcome({ onStart, resume }: Props) {
  return <Landing onNext={onStart} resume={resume} />
}
