export type ActiveTab =
  | "episodes"
  | "annotations"
  | "statistics"
  | "frames"
  | "insights"
  | "filtering"
  | "doctor"
  | "urdf";

const ACTIVE_TABS: readonly ActiveTab[] = [
  "episodes",
  "annotations",
  "statistics",
  "frames",
  "insights",
  "filtering",
  "doctor",
  "urdf",
];

export function isActiveTab(value: string | null): value is ActiveTab {
  return ACTIVE_TABS.includes(value as ActiveTab);
}

export function buildTabUrlSearch(
  currentSearch: string,
  tab: ActiveTab,
): string {
  const search = currentSearch.startsWith("?")
    ? currentSearch.slice(1)
    : currentSearch;
  const params = new URLSearchParams(search);
  params.set("tab", tab);
  const nextSearch = params.toString();
  return nextSearch ? `?${nextSearch}` : "";
}
