import time
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


from misc_helpers import display_secs_as_time, get_behavior_idxs_by_trial_idx


def plot_session_log(nwbfile):
    """
    Plot various values (like target posiion, cursor position, etc.) over time
    throughout a session.

    :param NWBFile nwbfile: An NWBFile, as returned by `NWBHDF5IO.read`.
    """
    trial_start_times = np.array(nwbfile.trials["start_time"])
    target_x_positions = np.array([p[0] for p in nwbfile.trials["target_pos"]])
    target_y_positions = np.array([p[1] for p in nwbfile.trials["target_pos"]])

    fig, ax = plt.subplots()

    ax.step(trial_start_times, target_x_positions, label="target position (x)")
    ax.step(trial_start_times, target_y_positions, label="target position (y)")

    ax.set_xlabel("time (hh:mm)")
    ax.set_ylabel("distance (m)")
    ax.set_title("Session log")
    ax.xaxis.set_major_formatter(lambda secs, _: display_secs_as_time(secs))
    ax.legend()

    plt.show()


def plot_target_positions(nwbfile):
    """
    Plot all target positions, in space.

    :param NWBFile nwbfile: An NWBFile, as returned by `NWBHDF5IO.read`.
    """
    unique_target_points = set((x, y) for x, y, _ in nwbfile.trials["target_pos"])
    x_vals, y_vals = zip(*unique_target_points)

    fig, ax = plt.subplots()

    ax.scatter(x_vals, y_vals)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Target positions")

    plt.show()


def inspect_trials_columns(nwbfile):
    """
    Print information about each column available in trials.

    :param NWBFile nwbfile: An NWBFile, as returned by `NWBHDF5IO.read`.
    """
    for colname in nwbfile.trials.colnames:
        if colname == "timeseries":
            continue
        print("=====")
        print(f"{colname}: {len(nwbfile.trials[colname])}")
        print(nwbfile.trials[colname][:10])
        print(nwbfile.trials[colname][-10:])


def display_event_timeline(sorted_events):
    """
    Display trial events in a pretty columnar printout, with times and time differences.

    :param dict[] sorted_events: Each dict in the list has fields "timestamp" and
        "field". The timestamp is seconds into the session, and the field is one of
        "start_time", "go_cue_time", etc. as they appear in the trials structure of the
        nwb file.
    """
    prev_time = 0
    for event_dict in sorted_events[:50]:
        t = event_dict["timestamp"]
        f = event_dict["field"]
        print("{:>10f} {:>20s} (+{:>})".format(t, f, round(t - prev_time, 6)))
        prev_time = t


def plot_trial_hand_trajectories(nwbfile, trial_idxs=(3, 4, 5, 6)):
    """
    Create gifs of hand trajectories overlaid on the target configuration.

    :param NWBFile nwbfile: An NWBFile, as returned by `NWBHDF5IO.read`.
    """

    hand_positions = nwbfile.processing["behavior"]["Position"]["Hand"]

    behavior_idxs_by_trial_idx = get_behavior_idxs_by_trial_idx(nwbfile)

    unique_target_points = set((x, y) for x, y, _ in nwbfile.trials["target_pos"])
    all_target_x_vals, all_target_y_vals = zip(*unique_target_points)

    trial_target_positions = nwbfile.trials["target_pos"]
    for trial_idx in trial_idxs:
        target_pos = trial_target_positions[trial_idx]
        behavior_idxs = behavior_idxs_by_trial_idx[trial_idx]
        start_idx = behavior_idxs["start_idx"]
        stop_idx = behavior_idxs["stop_idx"]
        trial_hand_positions = hand_positions.data[start_idx:stop_idx]
        trial_hand_x = trial_hand_positions[:, 0]
        trial_hand_y = trial_hand_positions[:, 1]

        # Make an animation of the trial.
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(all_target_x_vals, all_target_y_vals, color="red")
        ax.scatter([target_pos[0]], [target_pos[1]], color="green")
        (trajectory,) = ax.plot([], [])
        ax.set_xlim([-200, 200])
        ax.set_ylim([-200, 200])
        ax.set_aspect("equal", adjustable="box")

        trial_num_ms = stop_idx - start_idx
        ms_per_frame = 50
        anim_num_frames = math.floor(trial_num_ms / ms_per_frame)

        def animate(frame):
            num_ms_so_far = frame * ms_per_frame
            trajectory.set_data(
                trial_hand_x[:num_ms_so_far], trial_hand_y[:num_ms_so_far]
            )

        anim = FuncAnimation(
            fig,
            animate,
            frames=anim_num_frames,
            interval=ms_per_frame,
            repeat=False,
        )

        fps = 1000 / ms_per_frame
        anim.save(f"trial_{trial_idx}_trajectory.gif", writer="pillow", fps=fps)
