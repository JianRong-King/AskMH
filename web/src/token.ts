// Session token storage, kept separate from api.ts so the two don't import each other.
//
// localStorage rather than an httpOnly cookie: it is simpler across the dev/prod origin
// split, and this token grants nothing beyond demo chat access.

const KEY = 'askmh-token'

export function getToken(): string | null {
  try {
    return localStorage.getItem(KEY)
  } catch {
    return null
  }
}

export function setToken(token: string) {
  try {
    localStorage.setItem(KEY, token)
  } catch {
    // storage blocked: the session just won't survive a reload
  }
}

export function clearToken() {
  try {
    localStorage.removeItem(KEY)
  } catch {
    // nothing to do
  }
}
