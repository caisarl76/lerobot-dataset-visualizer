# Task 14 Annotation Tab URL Design

## Context

The Task 14 browser gate opens
`/local/pnp_trash/episode_0?tab=annotations`. Live verification showed that the
viewer ignores `tab`, restores the previous `sessionStorage` value, and opens
the Episodes tab. The video and chart data load correctly, but the documented
URL does not reach the task-index curation workspace.

## Decision

Resolve the initial viewer tab with this precedence:

1. one non-repeated, recognized, available `tab` query value;
2. a recognized, available `sessionStorage.activeTab` value;
3. `episodes`.

Query values may name any member of the existing `ActiveTab` union, including
`doctor`. Persisted values intentionally retain the narrower legacy whitelist,
which omits `doctor`; this preserves the existing session-restoration contract
rather than silently turning the URL fix into a second behavior change.
Unknown, empty, or repeated query values do not become application state.

Availability is evaluated for both query and persisted values. `urdf` is
available only when `hasURDFSupport(robotType)` is true and
`codebaseVersion >= "v3.0"`; otherwise it is ignored and resolution continues
to the next source. This closes the existing stale-session edge case as well as
preventing `?tab=urdf` from entering an unavailable, hidden state.

This change is initial-load authority only: tab clicks retain the existing
state and session persistence behavior, and no new router writes are
introduced. The URL may therefore become stale after a click, and same-mount
back/forward changes to `tab` are not applied. Those limitations are deliberate
for this narrow Task 14 repair.

The implementation will use a small pure resolver so precedence and invalid
input behavior can be tested without mounting the full episode viewer.

## Boundaries

- No curation API, source-data, video, timeline, authentication, or proxy
  contract changes.
- No Task 15 worker, batch, Cosmos proposal, or review action.
- The `local/pnp_trash` plus `v2.1` dataset predicate remains the sole reason
  the Annotations tab defaults to task-index mode.
- URLs without `tab` retain legacy session restoration, except that an
  unavailable persisted `urdf` value is rejected using the existing URDF
  visibility predicate.

## Verification

Test-first resolver coverage will prove query precedence, the legacy persisted
whitelist, default fallback, repeated-query rejection, and eligible/ineligible
URDF behavior for both query and persisted inputs. A separate
initialization-path component test will render the real episode viewer with
controlled dataset metadata, URL, and session storage; it must prove the viewer
actually uses the resolver and reaches Annotations for `?tab=annotations`.
That wiring test also covers a repeated query and eligible/ineligible URDF
cases from both authority sources.

After the frontend suite passes, the live Task 14 browser gate must prove that
the documented URL renders the task-index workspace, seven-phase timeline,
metadata charts, and a seekable video. Browser-visible DOM, loaded scripts, and
local-asset request headers must contain neither ephemeral runtime secret nor
Cosmos endpoint configuration; local assets must continue to load directly
from port 8000.
