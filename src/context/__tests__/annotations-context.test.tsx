import {
  afterEach,
  beforeEach,
  describe,
  expect,
  mock,
  spyOn,
  test,
} from "bun:test";
import { act, renderHook } from "@testing-library/react";

import type { LanguageAtom } from "../../types/language.types";
import * as client from "../../utils/annotationsClient";
import { AnnotationsProvider, useAnnotations } from "../annotations-context";

const ident = { repoId: "test/annotation-hydration", revision: "draft" };
const storageKey = (episode: number) =>
  `lerobot-annotations:v2:${ident.repoId}@${ident.revision}::${episode}`;

function atom(content: string): LanguageAtom {
  return {
    role: "assistant",
    content,
    style: "subtask",
    timestamp: 0,
    camera: null,
    tool_calls: null,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

let remote: Map<number, ReturnType<typeof deferred<LanguageAtom[]>>>;
let timestamps: Map<number, ReturnType<typeof deferred<number[]>>>;
let saved: ReturnType<typeof deferred<{ path: string | null }>>;

beforeEach(() => {
  remote = new Map(
    [0, 1].map((episode) => [episode, deferred<LanguageAtom[]>()]),
  );
  timestamps = new Map(
    [0, 1].map((episode) => [episode, deferred<number[]>()]),
  );
  saved = deferred<{ path: string | null }>();
  spyOn(client, "isAnnotateBackendEnabled").mockReturnValue(true);
  spyOn(client, "fetchEpisodeAtomsWithHash").mockImplementation(
    async (episode) => ({ atoms: await remote.get(episode)!.promise }),
  );
  spyOn(client, "fetchFrameTimestamps").mockImplementation(
    (episode) => timestamps.get(episode)!.promise,
  );
  spyOn(client, "saveEpisodeAtoms").mockImplementation(() => saved.promise);
});

afterEach(() => {
  // Restore real client exports; no mock.module overrides or environment changes.
  mock.restore();
});

describe("AnnotationsProvider hydration and save races", () => {
  for (const cached of [false, true]) {
    test(`a saved remote empty episode replaces stale ${cached ? "session snapshot" : "parquet"} atoms`, async () => {
      const stale = [atom("Stale saved instruction")];
      if (cached)
        sessionStorage.setItem(
          storageKey(0),
          JSON.stringify({
            atoms: stale,
            savedSnapshot: JSON.stringify(stale),
          }),
        );
      const { result } = renderHook(useAnnotations, {
        wrapper: AnnotationsProvider,
      });
      act(() => result.current.setEpisode(0, ident, stale));
      expect(result.current.atoms).toEqual(stale);
      expect(result.current.dirty).toBe(false);

      await act(async () => {
        remote.get(0)!.resolve([]);
      });

      expect(result.current.atoms).toEqual([]);
      expect(result.current.dirty).toBe(false);
      expect(JSON.parse(sessionStorage.getItem(storageKey(0))!)).toEqual({
        atoms: [],
        savedSnapshot: "[]",
      });
    });
  }

  for (const draft of [[], [atom("Unsaved local instruction")]]) {
    test(`a cached unsaved ${draft.length ? "nonempty" : "empty"} draft survives remote hydration`, async () => {
      const oldSaved = [atom("Old baseline")];
      const remoteSaved = [atom("New remote baseline")];
      sessionStorage.setItem(
        storageKey(0),
        JSON.stringify({
          atoms: draft,
          savedSnapshot: JSON.stringify(oldSaved),
        }),
      );
      const { result } = renderHook(useAnnotations, {
        wrapper: AnnotationsProvider,
      });
      act(() => result.current.setEpisode(0, ident, oldSaved));
      expect(result.current.dirty).toBe(true);

      await act(async () => {
        remote.get(0)!.resolve(remoteSaved);
      });

      expect(result.current.atoms).toEqual(draft);
      expect(result.current.dirty).toBe(true);
      // The new remote baseline, rather than the stale parquet seed, governs dirtiness.
      act(() => {
        result.current.resetAtoms();
        result.current.addAtoms(remoteSaved);
      });
      expect(result.current.dirty).toBe(false);
    });
  }

  test("deleting all atoms while hydration is pending keeps the unsaved empty draft", async () => {
    const initial = [atom("Original")];
    const { result } = renderHook(useAnnotations, {
      wrapper: AnnotationsProvider,
    });
    act(() => result.current.setEpisode(0, ident, initial));
    act(() => result.current.resetAtoms());
    await act(async () => {
      remote.get(0)!.resolve([atom("Remote saved instruction")]);
    });
    expect(result.current.atoms).toEqual([]);
    expect(result.current.dirty).toBe(true);
  });

  test("late old-episode atoms and timestamps cannot overwrite the new episode", async () => {
    const current = [atom("Episode one")];
    const { result } = renderHook(useAnnotations, {
      wrapper: AnnotationsProvider,
    });
    act(() =>
      result.current.setEpisode(0, ident, [atom("Episode zero")], [0, 0.1]),
    );
    act(() => result.current.setEpisode(1, ident, [], [0, 0.2]));
    await act(async () => {
      remote.get(1)!.resolve(current);
      timestamps.get(1)!.resolve([0, 0.25]);
    });
    await act(async () => {
      remote.get(0)!.resolve([atom("Late stale response")]);
      timestamps.get(0)!.resolve([0, 0.125]);
    });
    expect(result.current.episodeId).toBe(1);
    expect(result.current.atoms).toEqual(current);
    expect(result.current.frameTimestamps).toEqual([0, 0.25]);
    expect(result.current.dirty).toBe(false);
  });

  test("edits made during a save remain dirty after that save succeeds", async () => {
    const { result } = renderHook(useAnnotations, {
      wrapper: AnnotationsProvider,
    });
    act(() => result.current.setEpisode(0, ident));
    await act(async () => {
      remote.get(0)!.resolve([]);
    });
    const submitted = atom("Submitted instruction");
    const laterEdit = atom("Edit after request started");
    act(() => result.current.addAtom(submitted));
    let saving!: ReturnType<typeof result.current.save>;
    act(() => {
      saving = result.current.save();
    });
    expect(result.current.saving).toBe(true);
    expect(client.saveEpisodeAtoms).toHaveBeenCalledWith(0, ident, [submitted]);
    act(() => result.current.addAtom(laterEdit));
    await act(async () => {
      saved.resolve({ path: "/draft/meta/lerobot_annotations.json" });
      expect(await saving).toEqual({
        ok: true,
        path: "/draft/meta/lerobot_annotations.json",
      });
    });
    expect(result.current.atoms).toEqual([submitted, laterEdit]);
    expect(result.current.dirty).toBe(true);
    expect(result.current.saving).toBe(false);
    expect(
      JSON.parse(sessionStorage.getItem(storageKey(0))!).savedSnapshot,
    ).toBe(JSON.stringify([submitted]));
    // Only the submitted version was saved, so removing the later edit is clean.
    act(() => result.current.deleteAtom(laterEdit));
    expect(result.current.dirty).toBe(false);
  });

  test("an old episode's save completion cannot mark the new episode saved", async () => {
    const { result } = renderHook(useAnnotations, {
      wrapper: AnnotationsProvider,
    });
    act(() => result.current.setEpisode(0, ident));
    await act(async () => {
      remote.get(0)!.resolve([]);
    });
    act(() => result.current.addAtom(atom("Old episode edit")));
    let saving!: ReturnType<typeof result.current.save>;
    act(() => {
      saving = result.current.save();
    });
    act(() => result.current.setEpisode(1, ident));
    await act(async () => {
      remote.get(1)!.resolve([]);
    });
    const newEdit = atom("New episode edit");
    act(() => result.current.addAtom(newEdit));
    await act(async () => {
      saved.resolve({ path: "/old/meta/lerobot_annotations.json" });
      expect(await saving).toEqual({
        ok: false,
        error: "Episode changed during save",
      });
    });
    expect(result.current.episodeId).toBe(1);
    expect(result.current.atoms).toEqual([newEdit]);
    expect(result.current.dirty).toBe(true);
    expect(result.current.saving).toBe(false);
    expect(
      JSON.parse(sessionStorage.getItem(storageKey(1))!).savedSnapshot,
    ).toBe("[]");
  });
});

test("saves carry the loaded hash and advance to the acknowledged hash", async () => {
  spyOn(client, "fetchEpisodeAtomsWithHash").mockResolvedValue({
    atoms: [],
    annotation_sha256: "old",
  });
  const request = spyOn(client, "saveEpisodeAtoms").mockResolvedValue({
    path: "/draft",
    annotation_sha256: "new",
  });
  const { result } = renderHook(useAnnotations, {
    wrapper: AnnotationsProvider,
  });
  await act(async () => {
    result.current.setEpisode(0, ident);
  });
  act(() => result.current.addAtom(atom("First")));
  await act(async () => {
    expect((await result.current.save()).ok).toBe(true);
  });
  expect(request.mock.calls[0][3]).toBe("old");
  act(() => result.current.addAtom(atom("Second")));
  await act(async () => {
    await result.current.save();
  });
  expect(request.mock.calls[1][3]).toBe("new");
});
test("a conflict preserves the unsaved draft and stale cached hash", async () => {
  const draft = [atom("My draft")];
  sessionStorage.setItem(
    storageKey(0),
    JSON.stringify({
      atoms: draft,
      savedSnapshot: "[]",
      annotationSha256: "cached-old",
    }),
  );
  spyOn(client, "fetchEpisodeAtomsWithHash").mockResolvedValue({
    atoms: [atom("Other tab")],
    annotation_sha256: "remote-new",
  });
  const request = spyOn(client, "saveEpisodeAtoms").mockRejectedValue(
    new Error("Annotations changed; reload before saving"),
  );
  const { result } = renderHook(useAnnotations, {
    wrapper: AnnotationsProvider,
  });
  await act(async () => {
    result.current.setEpisode(0, ident);
  });
  await act(async () => {
    expect((await result.current.save()).ok).toBe(false);
  });
  expect(request.mock.calls[0][3]).toBe("cached-old");
  expect(result.current.atoms).toEqual(draft);
  expect(result.current.dirty).toBe(true);
});
test("saving before server hydration cannot overwrite another tab", async () => {
  const request = spyOn(client, "saveEpisodeAtoms");
  const { result } = renderHook(useAnnotations, {
    wrapper: AnnotationsProvider,
  });
  act(() => result.current.setEpisode(0, ident));
  act(() => result.current.addAtom(atom("Pending hydration")));
  await act(async () => {
    expect((await result.current.save()).ok).toBe(false);
  });
  expect(request).not.toHaveBeenCalled();
});
