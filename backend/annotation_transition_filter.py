"""Detect mode transitions whose motion starts after a short pause."""

from __future__ import annotations

from typing import Any

import numpy as np


LEG_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


UPPER_BODY_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
) + tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
)
BODY_JOINT_NAMES = LEG_JOINT_NAMES + UPPER_BODY_JOINT_NAMES
PLANNER_MODES = (2, 3)


def detect_transition_intervals(
    stream_modes: Any,
    joint_states: Any,
    fps: Any,
    joint_names: Any,
) -> list[dict[str, int]]:
    """Return pauses after Pose ↔ Planner switches, bounded by the next switch.

    Planner entry uses leg motion. Pose entry accepts motion in any body joint,
    so resumed manipulation with stationary legs is retained.
    """
    try:
        rate = float(fps)
        modes = np.asarray(stream_modes)
        states = np.asarray(joint_states)
        names = list(joint_names)
    except (TypeError, ValueError, OverflowError):
        return []
    if modes.ndim == 2 and modes.shape[1] == 1:
        modes = modes[:, 0]
    if isinstance(fps, bool):
        return []
    if not np.isfinite(rate) or rate <= 0 or modes.ndim != 1 or states.ndim != 2:
        return []
    if len(modes) != len(states) or len(names) != states.shape[1]:
        return []
    if not np.issubdtype(states.dtype, np.number):
        return []
    try:
        if not np.isfinite(states).all() or not np.isfinite(modes).all():
            return []
    except TypeError:
        return []
    leg_columns = [names.index(name) for name in LEG_JOINT_NAMES if name in names]
    body_columns = [names.index(name) for name in BODY_JOINT_NAMES if name in names]

    window = max(2, round(0.2 * rate))
    n = len(modes)
    result: list[dict[str, int]] = []
    for switch in range(1, n):
        previous, current = modes[switch - 1], modes[switch]
        if (
            previous == 1
            and current in PLANNER_MODES
            and len(leg_columns) == len(LEG_JOINT_NAMES)
        ):
            columns = leg_columns
        elif (
            previous in PLANNER_MODES
            and current == 1
            and len(body_columns) == len(BODY_JOINT_NAMES)
        ):
            columns = body_columns
        else:
            continue
        stop = switch + 1
        while stop < n and modes[stop] == current:
            stop += 1
        # Three consecutive sliding windows, each wholly in the destination-mode run.
        last_start = stop - window
        onset = None
        for start in range(switch, last_start + 1):
            if start + 2 > last_start:
                break
            qualifies = True
            for offset in range(3):
                values = states[start + offset : start + offset + window, columns]
                if (
                    float((np.max(values, axis=0) - np.min(values, axis=0)).max())
                    <= 0.1
                ):
                    qualifies = False
                    break
            if qualifies:
                onset = start
                break
        if onset is not None and onset > switch:
            result.append({"start_frame": switch, "end_frame": onset})
    return result
