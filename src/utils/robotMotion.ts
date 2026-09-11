/** Hold the last measured pose; never stretch timestamps to video duration. */
export function robotFrameAt(timestamps: number[], time: number): number {
  let lo = 0;
  let hi = timestamps.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    // Float32 parquet timestamps and video clock arithmetic differ by microseconds.
    if (timestamps[mid] <= time + 1e-5) lo = mid + 1;
    else hi = mid;
  }
  return Math.max(0, lo - 1);
}

/** Recognize G1 recordings whose collector omitted robot_type metadata. */
export function hasG1JointNames(names: string[] = []): boolean {
  return [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
  ].every((name) => names.includes(name));
}
