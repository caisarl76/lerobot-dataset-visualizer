export const ACTIVE_TABS = [
  "episodes",
  "annotations",
  "statistics",
  "frames",
  "insights",
  "filtering",
  "doctor",
  "urdf",
] as const;

export type ActiveTab = (typeof ACTIVE_TABS)[number];

const QUERY_TABS = new Set<string>(ACTIVE_TABS);
const LEGACY_PERSISTED_TABS = new Set<string>([
  "episodes",
  "annotations",
  "statistics",
  "frames",
  "insights",
  "filtering",
  "urdf",
]);

export interface EpisodeViewerTabAvailability {
  urdfAvailable: boolean;
}

export interface EpisodeViewerTabSearchParams {
  getAll(name: string): string[];
}

function isAvailableTab(
  value: string,
  availability: EpisodeViewerTabAvailability,
): value is ActiveTab {
  return (
    QUERY_TABS.has(value) && (value !== "urdf" || availability.urdfAvailable)
  );
}

export function resolveInitialEpisodeViewerTab(
  searchParams: EpisodeViewerTabSearchParams,
  persisted: string | null,
  availability: EpisodeViewerTabAvailability,
): ActiveTab {
  const queryValues = searchParams.getAll("tab");
  if (
    queryValues.length === 1 &&
    isAvailableTab(queryValues[0], availability)
  ) {
    return queryValues[0];
  }
  if (
    persisted !== null &&
    LEGACY_PERSISTED_TABS.has(persisted) &&
    isAvailableTab(persisted, availability)
  ) {
    return persisted;
  }
  return "episodes";
}
