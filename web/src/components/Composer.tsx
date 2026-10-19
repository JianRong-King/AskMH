import { useEffect, useRef, useState, type FormEvent } from 'react'

interface Props {
  disabled: boolean
  onSend: (text: string) => void
}

export default function Composer({ disabled, onSend }: Props) {
  const [text, setText] = useState('')
  const input = useRef<HTMLInputElement>(null)

  // Hand focus back once the reply arrives, so the user can keep typing.
  useEffect(() => {
    if (!disabled) input.current?.focus()
  }, [disabled])

  function submit(e: FormEvent) {
    e.preventDefault()
    if (!text.trim() || disabled) return
    onSend(text)
    setText('')
  }

  return (
    <form onSubmit={submit} className="flex gap-2">
      <input
        ref={input}
        value={text}
        onChange={e => setText(e.target.value)}
        disabled={disabled}
        maxLength={2000}
        placeholder="Ask about baggage, check-in, Enrich, refunds..."
        aria-label="Message"
        className="flex-1 rounded-full border border-slate-300 bg-white px-4 py-2 outline-none focus:border-mh-navy focus:ring-2 focus:ring-mh-navy/10 disabled:bg-slate-100"
      />
      <button
        type="submit"
        disabled={disabled || !text.trim()}
        className="rounded-full bg-mh-navy px-5 py-2 font-medium text-white hover:bg-mh-navy-light disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        Send
      </button>
    </form>
  )
}
