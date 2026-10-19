import { useCallback, useEffect, useRef, useState } from 'react'
import { getConfig, loginWithGoogle, type AppConfig } from './api'
import { clearToken, getToken, setToken } from './token'

const GIS_SRC = 'https://accounts.google.com/gsi/client'

// Minimal shape of the bits of Google Identity Services we use.
interface GoogleId {
  initialize(opts: { client_id: string; callback: (r: { credential: string }) => void }): void
  renderButton(parent: HTMLElement, opts: Record<string, unknown>): void
}
declare global {
  interface Window {
    google?: { accounts: { id: GoogleId } }
  }
}

let scriptPromise: Promise<void> | null = null

function loadGis(): Promise<void> {
  if (scriptPromise) return scriptPromise
  scriptPromise = new Promise((resolve, reject) => {
    if (window.google?.accounts?.id) return resolve()
    const el = document.createElement('script')
    el.src = GIS_SRC
    el.async = true
    el.onload = () => resolve()
    el.onerror = () => reject(new Error('Could not load Google sign-in'))
    document.head.appendChild(el)
  })
  return scriptPromise
}

export type AuthState = 'loading' | 'signed-out' | 'ready' | 'unreachable'

/** Resolves whether sign-in is required, and drives the Google flow when it is. */
export function useAuth() {
  const [config, setConfig] = useState<AppConfig | null>(null)
  const [state, setState] = useState<AuthState>('loading')
  const [error, setError] = useState('')
  const buttonRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    getConfig()
      .then(cfg => {
        setConfig(cfg)
        // Auth off (local dev) means everyone is the anonymous user, so skip straight in.
        setState(!cfg.auth_enabled || getToken() ? 'ready' : 'signed-out')
      })
      .catch(() => setState('unreachable'))
  }, [])

  const onCredential = useCallback(async (credential: string) => {
    try {
      const res = await loginWithGoogle(credential)
      setToken(res.token)
      setState('ready')
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Sign-in failed')
    }
  }, [])

  // Render Google's button once we know we need it and the container exists.
  useEffect(() => {
    if (state !== 'signed-out' || !config?.google_client_id || !buttonRef.current) return
    let cancelled = false
    loadGis()
      .then(() => {
        if (cancelled || !buttonRef.current || !window.google) return
        window.google.accounts.id.initialize({
          client_id: config.google_client_id,
          callback: r => void onCredential(r.credential),
        })
        window.google.accounts.id.renderButton(buttonRef.current, {
          theme: 'outline',
          size: 'large',
          text: 'signin_with',
        })
      })
      .catch(e => setError(e.message))
    return () => {
      cancelled = true
    }
  }, [state, config, onCredential])

  const signOut = useCallback(() => {
    clearToken()
    setState('signed-out')
  }, [])

  return { state, config, error, buttonRef, signOut }
}
