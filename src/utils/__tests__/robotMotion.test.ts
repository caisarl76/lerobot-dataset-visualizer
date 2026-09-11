import { describe, expect, test } from "bun:test";
import { robotFrameAt, hasG1JointNames } from "../robotMotion";

describe("measured robot frame selection", () => {
  test("uses recorded irregular timestamps without stretching or anticipating a pose", () => {
    const times = [0, 0.02, 0.055, 0.08];
    expect(robotFrameAt(times, 0.054)).toBe(1);
    expect(robotFrameAt(times, 0.055)).toBe(2);
    expect(robotFrameAt(times, 0.079)).toBe(2);
  });
  test("clamps outside the recorded range and supports backward/exclusion seeks", () => {
    const times = [0, 0.02, 0.04, 0.06];
    expect(robotFrameAt(times, 10)).toBe(3);
    expect(robotFrameAt(times, 0.01)).toBe(0);
    expect(robotFrameAt(times, -1)).toBe(0);
    expect(robotFrameAt(times, 0.06)).toBe(3);
    expect(robotFrameAt([0], 20)).toBe(0);
  });
});

test("recognizes unnamed G1 recordings without treating arm-only recordings as G1", () => {
  expect(hasG1JointNames()).toBe(false);
  expect(hasG1JointNames(["left_wrist_yaw_joint"])).toBe(false);
  expect(
    hasG1JointNames([
      "left_hip_pitch_joint",
      "right_hip_pitch_joint",
      "waist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint",
      "left_ankle_roll_joint",
      "right_ankle_roll_joint",
      "left_wrist_yaw_joint",
      "right_wrist_yaw_joint",
    ]),
  ).toBe(true);
});

test("frame boundaries tolerate float32 storage and segmented-video subtraction", () => {
  expect(robotFrameAt([0, 0.10000000149011612], 0.1)).toBe(1);
  expect(robotFrameAt([0, 1], 1.9 - 0.9)).toBe(1);
});
