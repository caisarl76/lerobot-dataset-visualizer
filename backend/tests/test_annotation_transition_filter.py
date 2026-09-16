import numpy as np
import pytest

from annotation_transition_filter import detect_transition_intervals


NAMES = [
    "right_knee_joint",
    "left_ankle_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_ankle_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_hip_pitch_joint",
    "right_hip_yaw_joint",
]


def data(n=30, fps=50):
    return np.zeros((n, len(NAMES))), fps


def test_detects_pause_and_onset_with_permuted_names():
    states, fps = data(40)
    modes = np.array([1] * 5 + [3] * 35)
    states[25:, NAMES.index("left_knee_joint")] = 0.2
    assert detect_transition_intervals(modes[:, None], states, fps, NAMES) == [
        {"start_frame": 5, "end_frame": 16}
    ]


def test_switch_back_limits_search_and_no_motion_is_skipped():
    states, fps = data(30)
    modes = np.array([1] * 3 + [3] * 8 + [1] * 19)
    states[3:11, 0] = 1
    assert detect_transition_intervals(modes, states, fps, NAMES) == []


def test_ignored_transitions_missing_names_and_invalid_inputs():
    states, fps = data(40)
    modes = np.array([2] * 5 + [3] * 35)
    assert detect_transition_intervals(modes, states, fps, NAMES) == []
    assert detect_transition_intervals(modes, states, fps, NAMES[:-1]) == []
    assert detect_transition_intervals(modes, states, 0, NAMES) == []
    states[0, 0] = np.nan
    assert detect_transition_intervals(modes, states, fps, NAMES) == []


def test_thirty_fps_uses_six_frame_windows():
    states, fps = data(30, 30)
    modes = np.array([1] * 3 + [3] * 27)
    states[15:, 0] = 0.2
    assert detect_transition_intervals(modes, states, fps, NAMES) == [
        {"start_frame": 3, "end_frame": 10}
    ]


def test_stationary_mode_run_and_motion_at_switch_are_not_excluded():
    states, fps = data(40)
    modes = np.array([1] * 5 + [3] * 35)
    assert detect_transition_intervals(modes, states, fps, NAMES) == []
    states[:, 0] = np.arange(40) * 0.1
    assert detect_transition_intervals(modes, states, fps, NAMES) == []


UPPER_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + [
    f"{side}_{joint}_joint"
    for side in ["left", "right"]
    for joint in [
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    ]
]


@pytest.mark.parametrize("planner", [2, 3])
def test_planner_to_pose_detects_arm_resume_with_stationary_legs(planner):
    names = NAMES + UPPER_NAMES
    states = np.zeros((40, len(names)))
    states[25:, names.index("left_elbow_joint")] = 0.2
    modes = [planner] * 5 + [1] * 35
    assert detect_transition_intervals(modes, states, 50, names) == [
        {"start_frame": 5, "end_frame": 16}
    ]


def test_reverse_does_not_exclude_resumed_leg_motion_or_cross_next_switch():
    names = NAMES + UPPER_NAMES
    states = np.zeros((50, len(names)))
    states[:, 0] = np.arange(50) * 0.1
    assert detect_transition_intervals([3] * 5 + [1] * 45, states, 50, names) == []
    states[:] = 0
    states[35:, names.index("left_elbow_joint")] = 0.2
    assert (
        detect_transition_intervals([3] * 5 + [1] * 20 + [3] * 25, states, 50, names)
        == []
    )


def test_pose_to_regular_planner_uses_leg_motion():
    states, fps = data(40)
    states[25:, 0] = 0.2
    assert detect_transition_intervals([1] * 5 + [2] * 35, states, fps, NAMES) == [
        {"start_frame": 5, "end_frame": 16}
    ]
