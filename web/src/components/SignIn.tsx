import type { RefObject } from 'react'

interface Props {
  buttonRef: RefObject<HTMLDivElement | null>
  error?: string
}

export default function SignIn({ buttonRef, error }: Props) {
  return (
    <div className="flex h-dvh items-center justify-center bg-gradient-to-br from-mh-navy via-mh-navy-light to-mh-navy px-4 text-slate-900">
      <div className="w-full max-w-sm rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-xl animate-rise">
        <h1 className="text-xl font-semibold text-mh-navy">AskMH</h1>
        <p className="mt-1 text-sm text-slate-500">Malaysia Airlines FAQ assistant</p>

        <p className="mt-6 text-sm text-slate-600">
          This demo runs on a shared free-tier model quota, so sign-in keeps it usable for
          everyone. Your Google account is used only to identify the session. Questions are logged without
          your account details to improve answers.
        </p>

        <div ref={buttonRef} className="mt-6 flex justify-center" />

        {error && <p className="mt-4 text-sm text-red-600">{error}</p>}

        <p className="mt-6 text-xs text-slate-400">
          Portfolio project. Content belongs to Malaysia Airlines.
        </p>
      </div>
    </div>
  )
}
