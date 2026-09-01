// Client-side helpers for the HF OAuth flow. The token is owned by
// AuthProvider (src/context/auth-context.tsx); these read-only helpers exist
// so non-React fetch utilities can attach `Authorization: Bearer <token>`
// without going through React.

const STORAGE_KEY = "lerobot-viz-oauth";

export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) return null;
    const parsed = JSON.parse(stored) as { accessToken?: string };
    return parsed.accessToken ?? null;
  } catch {
    return null;
  }
}

export function isAuthenticatedHfDestination(
  destination: string | URL,
): boolean {
  try {
    const url = destination instanceof URL ? destination : new URL(destination);
    return url.protocol === "https:" && url.hostname === "huggingface.co";
  } catch {
    return false;
  }
}

export function authHeaders(destination: string | URL): Record<string, string> {
  if (!isAuthenticatedHfDestination(destination)) return {};
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export const AUTH_STORAGE_KEY = STORAGE_KEY;

// Native <video> elements can't carry an Authorization header. To play videos
// from private datasets, we route them through our same-origin /api/proxy
// endpoint, which reads the access token from an HttpOnly cookie set during
// sign-in and forwards the request to huggingface.co. Returns the original
// URL when running server-side or when the user is not signed in.
export function proxyHfUrl(url: string): string {
  if (typeof window === "undefined") return url;
  if (!getAuthToken()) return url;
  if (!isAuthenticatedHfDestination(url)) return url;
  const parsed = new URL(url);
  return `/api/proxy${parsed.pathname}${parsed.search}`;
}
