# Task 14 Annotation Tab URL Design

## Context

The Task 14 browser gate opens
`/local/pnp_trash/episode_0?tab=annotations`. Live verification showed that the
viewer ignores `tab`, restores the previous `sessionStorage` value, and opens
the Episodes tab. The video and chart data load correctly, but the documented
URL does not reach the task-index curation workspace.

## Decision

Resolve the initial viewer tab with this precedence:

1. a recognized `tab` query value;
2. a recognized `sessionStorage.activeTab` value;
3. `episodes`.

Recognized values are exactly the existing `ActiveTab` union. Unknown, empty,
or repeated query values do not become application state. This change is
initial-load authority only: tab clicks retain the existing state and session
persistence behavior, and no new router writes are introduced.

The implementation will use a small pure resolver so precedence and invalid
input behavior can be tested without mounting the full episode viewer.

## Boundaries

- No curation API, source-data, video, timeline, authentication, or proxy
  contract changes.
- No Task 15 worker, batch, Cosmos proposal, or review action.
- The `local/pnp_trash` plus `v2.1` dataset predicate remains the sole reason
  the Annotations tab defaults to task-index mode.
- Existing URLs without `tab` retain their current session-restoration
  behavior.

## Verification

Test-first coverage will prove query precedence, session fallback, default
fallback, and rejection of invalid query/session values. After the frontend
suite passes, the live Task 14 browser gate must prove that the documented URL
renders the task-index workspace, seven-phase timeline, metadata charts, and a
seekable video. Browser-visible DOM, loaded scripts, and local-asset request
headers must contain neither ephemeral runtime secret nor Cosmos endpoint
configuration; local assets must continue to load directly from port 8000.
