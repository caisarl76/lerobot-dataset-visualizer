import React from "react";
import { afterEach, expect, spyOn, test } from "bun:test";
import { render, fireEvent, waitFor, cleanup } from "@testing-library/react";
import { TimeProvider, useTime } from "../../context/time-context";
import { SimpleVideosPlayer } from "../simple-videos-player";
import { AnnotationsProvider } from "../../context/annotations-context";

function Seek() {
  const { seek, setIsPlaying } = useTime();
  return (
    <button
      onClick={() => {
        setIsPlaying(false);
        seek(0.02);
      }}
    >
      Next frame
    </button>
  );
}
afterEach(cleanup);
test("a paused one-frame seek reaches the video, including its source-file offset", async () => {
  const play = spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue();
  const pause = spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(
    () => {},
  );
  try {
    const view = render(
      <TimeProvider duration={2}>
        <AnnotationsProvider>
          <SimpleVideosPlayer
            videosInfo={[
              {
                filename: "ego_view",
                url: "http://test/video.mp4",
                isSegmented: true,
                segmentStart: 0.9,
                segmentEnd: 2.9,
              },
            ]}
          />
          <Seek />
        </AnnotationsProvider>
      </TimeProvider>,
    );
    const video = view.container.querySelector("video")!;
    Object.defineProperty(video, "currentTime", {
      configurable: true,
      writable: true,
      value: 0,
    });
    fireEvent.loadedData(video);
    await waitFor(() => expect(video.currentTime).toBeCloseTo(0.9));
    fireEvent.click(view.getByText("Next frame"));
    await waitFor(() => expect(video.currentTime).toBeCloseTo(0.92));
  } finally {
    play.mockRestore();
    pause.mockRestore();
  }
});
