import React from "react";
import { afterEach, beforeEach, expect, mock, spyOn, test } from "bun:test";
import { act, fireEvent, render, waitFor } from "@testing-library/react";
import * as client from "../../utils/annotationsClient";
import {
  AnnotationsProvider,
  useAnnotations,
} from "../../context/annotations-context";
import { AnnotationReviewControl } from "../annotation-review-control";
const unreviewed = {
  status: "unreviewed" as const,
  annotation_sha256: "sha",
  exclusions_sha256: "clip-sha",
  reviewed_at: null,
  prediction_available: false,
};
const reviewed = {
  ...unreviewed,
  status: "reviewed" as const,
  reviewed_at: "now",
};
beforeEach(() => {
  spyOn(client, "isAnnotateBackendEnabled").mockReturnValue(true);
  spyOn(client, "fetchEpisodeAtomsWithHash").mockResolvedValue({
    atoms: [],
    annotation_sha256: "sha",
  });
  spyOn(client, "fetchFrameTimestamps").mockResolvedValue([]);
  spyOn(client, "fetchEpisodeReview").mockResolvedValue(unreviewed);
  spyOn(client, "setEpisodeReview").mockResolvedValue(reviewed);
});
afterEach(() => mock.restore());
function Editor({ episode }: { episode: number }) {
  const { setEpisode, addAtom } = useAnnotations();
  React.useEffect(() => {
    setEpisode(episode, { repoId: "org/data" }, []);
  }, [episode, setEpisode]);
  return (
    <>
      <button
        onClick={() =>
          addAtom({
            style: "subtask",
            role: "assistant",
            content: "x",
            timestamp: 0,
            camera: null,
            tool_calls: null,
          })
        }
      >
        Edit
      </button>
      <AnnotationReviewControl />
    </>
  );
}
const content = (episode = 1) => (
  <AnnotationsProvider>
    <Editor episode={episode} />
  </AnnotationsProvider>
);
test("requires click and disables while dirty", async () => {
  const ui = render(content());
  await ui.findByRole("button", { name: "Mark reviewed" });
  expect(client.setEpisodeReview).not.toHaveBeenCalled();
  fireEvent.click(ui.getByRole("button", { name: "Mark reviewed" }));
  await waitFor(() =>
    expect(client.setEpisodeReview).toHaveBeenCalledWith(
      1,
      { repoId: "org/data" },
      true,
      "sha",
      "clip-sha",
    ),
  );
  fireEvent.click(ui.getByRole("button", { name: "Edit" }));
  expect(ui.queryByRole("button", { name: "Mark reviewed" })).toBeNull();
});
test("discards stale response after navigation", async () => {
  let resolve!: (value: client.EpisodeReview) => void;
  spyOn(client, "setEpisodeReview").mockImplementationOnce(
    () =>
      new Promise((done) => {
        resolve = done;
      }),
  );
  const ui = render(content());
  await ui.findByRole("button", { name: "Mark reviewed" });
  fireEvent.click(ui.getByRole("button", { name: "Mark reviewed" }));
  ui.rerender(content(2));
  await act(async () => resolve({ ...reviewed, annotation_sha256: "old" }));
  expect(ui.queryByText("Reviewed")).toBeNull();
});
test("cannot review another tab's unseen annotations", async () => {
  spyOn(client, "fetchEpisodeReview").mockResolvedValue({
    ...unreviewed,
    annotation_sha256: "other-tab",
  });
  const ui = render(content());
  await ui.findByText(/Annotations changed in another tab/);
  expect(ui.queryByRole("button", { name: "Mark reviewed" })).toBeNull();
  expect(client.setEpisodeReview).not.toHaveBeenCalled();
});
test("cannot review until the displayed atoms have loaded with their hash", async () => {
  spyOn(client, "fetchEpisodeAtomsWithHash").mockImplementation(
    () => new Promise(() => {}),
  );
  const ui = render(content());
  await act(async () => {});
  expect(ui.queryByRole("button", { name: "Mark reviewed" })).toBeNull();
  expect(client.setEpisodeReview).not.toHaveBeenCalled();
});

test("reloads the review binding after exclusions change", async () => {
  const ui = render(content());
  await ui.findByRole("button", { name: "Mark reviewed" });
  spyOn(client, "fetchEpisodeReview").mockResolvedValue({
    ...unreviewed,
    exclusions_sha256: "new-clips",
  });
  await act(async () => {
    window.dispatchEvent(new window.Event("annotation-clipping-changed"));
  });
  fireEvent.click(await ui.findByRole("button", { name: "Mark reviewed" }));
  await waitFor(() =>
    expect(client.setEpisodeReview).toHaveBeenCalledWith(
      1,
      { repoId: "org/data" },
      true,
      "sha",
      "new-clips",
    ),
  );
});
