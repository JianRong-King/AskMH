export default function TypingIndicator() {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-500" role="status">
      <span className="flex gap-1">
        {[0, 150, 300].map(delay => (
          <span key={delay} className="h-2 w-2 animate-bounce rounded-full bg-slate-400"
                style={{ animationDelay: `${delay}ms` }} />
        ))}
      </span>
      AskMH is thinking...
    </div>
  )
}
