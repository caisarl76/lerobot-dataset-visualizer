import { describe, expect, test } from "bun:test";
import { buildTabUrlSearch } from "@/app/[org]/[dataset]/[episode]/tab-url";

describe("buildTabUrlSearch", () => {
  test("replaces an existing tab query param when switching tabs", () => {
    expect(buildTabUrlSearch("tab=urdf&t=4", "statistics")).toBe(
      "?tab=statistics&t=4",
    );
  });

  test("adds a tab query param when none exists", () => {
    expect(buildTabUrlSearch("", "frames")).toBe("?tab=frames");
  });
});
